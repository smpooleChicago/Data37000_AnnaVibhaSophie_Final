# cnn_animals10_from_filtered.py
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------------------------------------------
# 0. Device
# ------------------------------------------------------------
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
device = torch.device(device)
print("Using device:", device)

#%% [markdown]
# ------------------------------------------------------------
## 1. Data preparation from filtered CSVs
# ------------------------------------------------------------
#In this section we load the pre-filtered Open Images CSV files that contain only our target animal classes
#We define a manual dictionary `label_to_animal` that maps these Open Images label IDs (e.g. `/m/0bt9lr`) to human-readable animal names such as `"Dog"` or `"Cat"`
#We first filter the annotation tables to keep only rows whose label IDs appear in this dictionary, and then add a `ClassName` column with the corresponding animal name. 
#Finally, we merge the image lists with the filtered annotations on `ImageID` (for both train and test), so each row in `train_labels_merged` and `test_labels_merged` contains:
# (1) the image identifier and its metadata, 
# and (2) the associated label ID and animal class name. 

#%%
# Base directory you gave me

train_images_csv = f'.{os.sep}BigData/train_images.csv'
train_ann_csv    = f'.{os.sep}BigData/train_annotations.csv'
test_images_csv  = f'.{os.sep}BigData/test_images.csv'
test_ann_csv     = f'.{os.sep}BigData/test_annotations.csv'

train_labels = pd.read_csv(train_images_csv)   # ImageID + rotation + etc.
classes      = pd.read_csv(train_ann_csv)      # ImageID + LabelName + Confidence (for filtered animals)
["Dog", "Bird", "Horse", "Cat", "Bear","Sheep", "Cattle"]

# Your manual mapping MID -> animal name
label_to_animal = {
    "/m/0bt9lr": "Dog",
    "/m/015p6":  "Bird",
    "/m/03k3r":  "Horse",
    "/m/01yrx":  "Cat",
    "/m/01dws":  "Bear",
    "/m/07bgp":  "Sheep",
    "/m/01xq0k1": "Cattle"
}

# Keep only labels we know how to map
classes = classes[classes["LabelName"].isin(label_to_animal.keys())].copy()
classes["ClassName"] = classes["LabelName"].map(label_to_animal)

# Merge train-images file with annotations to attach LabelName + ClassName
train_labels_merged = train_labels.merge(
    classes[["ImageID", "LabelName", "ClassName"]],
    on="ImageID",
    how="inner"           # only keep images that have our animal label
)

print("Train labels merged shape:", train_labels_merged.shape)
print(train_labels_merged.head())

# Same for test split
test_images = pd.read_csv(test_images_csv)
test_ann    = pd.read_csv(test_ann_csv)
test_ann    = test_ann[test_ann["LabelName"].isin(label_to_animal.keys())].copy()
test_ann["ClassName"] = test_ann["LabelName"].map(label_to_animal)

test_labels_merged = test_images.merge(
    test_ann[["ImageID", "LabelName", "ClassName"]],
    on="ImageID",
    how="inner"
)

print("\nTest labels merged shape:", test_labels_merged.shape)
print(test_labels_merged.head())

#%% [markdown]
# ------------------------------------------------------------
# 2. Paths & label mapping (7 animal classes)
# ------------------------------------------------------------
#As Neural networks cannot work directly with string labels, we convert our animal classes into integer indices. 
#We first collect the set of distinct animal names that appear in `label_to_animal`, sort them, and assign each one a class index via the dictionary `class_to_idx` (e.g. `"Bear" -> 0`, `"Bird" -> 1`, etc.). 
#The list `idx_to_name` stores the reverse mapping from index back to class name, which is useful for decoding predictions.
#
#We then build a `mid_to_idx` dictionary that maps each Open Images label ID (MID) directly to its integer class index, by composing `label_to_animal` with `class_to_idx`. During dataset loading we will use `mid_to_idx` to convert the `LabelName` column (MIDs from the CSVs) into numerical labels suitable for `CrossEntropyLoss`. This ensures a consistent encoding of classes throughout training and evaluation.

# and prepare mappings for later use.

#%%
# Unique animal names (from your mapping)
animal_names = sorted(set(label_to_animal.values()))
class_to_idx = {name: i for i, name in enumerate(animal_names)}
idx_to_name  = animal_names

# MID -> class index using the animal_names ordering
mid_to_idx = {mid: class_to_idx[animal] for mid, animal in label_to_animal.items()}

print("\nAnimal names and indices:")
for name, i in class_to_idx.items():
    print(f"{i}: {name}")

print("\nMID -> class index mapping:")
for mid, idx in mid_to_idx.items():
    print(f"{mid:>10} -> {idx} ({idx_to_name[idx]})")
#every image's label can now be turned from a symbolic ID (MID) into a integer index
#that integer index is what PyTorch expercs for CrossEntropyLoss

#EX: mid_to_idx = {
    #"/m/0kpmf": 0,   # Dog
    #"/m/015p6": 1,   # Bird
    #"/m/01yrx": 2,   # Cat
    #...
#}
#idx_to_name = ["Dog", "Bird", "Cat", ...]  # position = class index




#%% [markdown]
# ------------------------------------------------------------
## 3. Custom Dataset and DataLoader construction
# ------------------------------------------------------------
# We define a custom PyTorch `Dataset` class `OpenImagesAnimalsDataset` that wraps the merged train/test DataFrames. 
# Each row contains an `ImageID` and a `LabelName` (MID). 
# The image is opened with PIL, converted to RGB, and passed through a torchvision transform pipeline to resize, augment, convert to a tensor, and normalize its pixel values.
# To obtain the target label for training, we read the MID from `row["LabelName"]` and convert it to an integer class index via `mid_to_idx[mid]`. 
# The dataset therefore returns pairs `(image_tensor, label_idx)`, where `image_tensor` has shape `[3, 64, 64]` and `label_idx` is an integer in `[0, num_classes)`. 
# Finally, we wrap the datasets in `DataLoader`s that handle batching, shuffling (for the training set), and parallel loading, producing mini-batches ready to be fed into the CNN.

TRAIN_IMG_DIR =  f'.{os.sep}train_images'
TEST_IMG_DIR  = f'.{os.sep}test_images'
#TRAIN_IMG_DIR = "/Users/annasirtori/.../FilteredImages_animalsProject/train_images"
#TEST_IMG_DIR  = "/Users/annasirtori/.../FilteredImages_animalsProject/test_images"


class OpenImagesAnimalsDataset(Dataset):
    def __init__(self, df, images_root, mid_to_idx, transform=None):
        """
        df: DataFrame with columns ['ImageID', 'LabelName', 'ClassName', ...]
        images_root: folder where <ImageID>.jpg/.jpeg/.png are stored
        mid_to_idx: dict mapping MID -> integer label index
        transform: torchvision transforms
        """
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.mid_to_idx = mid_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        # Try common extensions; adjust if needed.
        candidates = [
            os.path.join(self.images_root, f"{image_id}.jpg"),
            os.path.join(self.images_root, f"{image_id}.jpeg"),
            os.path.join(self.images_root, f"{image_id}.png"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not find image for ImageID={image_id}. Tried: {candidates}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageID"]
        mid      = row["LabelName"]              # e.g. "/m/0bt9lr"

        img_path = self._find_image_path(image_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = transforms.ToTensor()(img)

        # 🔥 This is exactly the logic you wrote:
        label_idx = self.mid_to_idx[mid]         # e.g. 0 for "Dog"
        return image_tensor, label_idx           # (3x64x64 tensor, 0)

#%%
# ------------------------------------------------------------
# 4. Transforms and DataLoaders
# ------------------------------------------------------------
IMG_SIZE = 64
#every image resized to 64x64 (target spatial resolution)

#train images: resize + augment + normalize
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

#for test images, only resize + normalize
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = OpenImagesAnimalsDataset(train_labels_merged, TRAIN_IMG_DIR, mid_to_idx, transform=transform_train)
test_dataset  = OpenImagesAnimalsDataset(test_labels_merged,  TEST_IMG_DIR,  mid_to_idx, transform=transform_test)

#wrap dataframes in the custom dataset object
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
# handles batching, shuffling, parallel loading
testloader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# sanity check
imgs, labels = next(iter(trainloader))
print("Batch images shape:", imgs.shape)   # [B, 3, 64, 64]
print("Batch labels shape:", labels.shape) # [B]

#%%
#%% [markdown]
# ------------------------------------------------------------
## 4b. Sample visualization
# ------------------------------------------------------------


#Utility function to vsualize some examples from a batch 
def show_sample_batch(images, labels, idx_to_name, num_samples=8):
    """
    Show a few images from a batch with their decoded class names.
    images: tensor [B, 3, H, W] (normalized)
    labels: tensor [B]
    """
    # unnormalize: x * std + mean; here mean=std=0.5 for all channels
    images = images * 0.5 + 0.5  # back to [0,1] approx

    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    if num_samples == 1:
        axes = [axes]

    #we will show num_samples images from the batch
    for i in range(num_samples):
        img_np = images[i].cpu().numpy()          # [3,H,W]
        img_np = np.transpose(img_np, (1, 2, 0))  # [H,W,3] for imshow
        axes[i].imshow(img_np)
        class_idx = labels[i].item()
        axes[i].set_title(idx_to_name[class_idx])
        axes[i].axis('off')

    plt.suptitle("Sample training images with labels")
    plt.tight_layout()
    plt.show()

show_sample_batch(imgs, labels, idx_to_name, num_samples=8)
#%%[markdown]
# ------------------------------------------------------------
# 5. CNN model 
# ------------------------------------------------------------
# We bulild a simple 2 convolutional blocks CNN architeture for classifying the 10 animal classes.
# Each block applies two `3×3` convolutions with ReLU activations, followed by a `2×2` max pooling layer that halves the spatial resolution. 
# Starting from a `3×64×64` RGB image, the first block produces `32×32×32` feature maps, and the second block produces `64×16×16` feature maps. 
# A `Dropout(0.25)` layer is added after the convolutional blocks to reduce overfitting by randomly zeroing a fraction of activations during training. 

#The fully connected classifier head (`self.fc_layers`) takes the final `64×16×16` feature maps and flattens them into a 16,384-dimensional feature vector per image. 
#A linear layer then reduces this to a 256-dimensional embedding, which is passed through a ReLU non-linearity and `Dropout(0.5)` for robust non-linear decision boundaries. 
#The final linear layer maps this 256-D representation to `num_classes` output logits, one per animal class. 
# During training, these logits are fed into `CrossEntropyLoss`, which applies a softmax and compares the resulting probability distribution to the true integer label index.

#Simple CNN with 2 conv blocks + FC layers: 
#each conv block: Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d
#then flatten + FC -> ReLU -> FC

#Block 1: 3→32 conv, ReLU, 32→32 conv, ReLU, then pool (64×64 → 32×32).
#Block 2: 32→64 conv, ReLU, 64→64 conv, ReLU, then pool (32×32 → 16×16).

class AnimalsCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Block 1: 3x64x64 -> 32x64x64 -> 32x64x64 -> MaxPool -> 32x32x32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32x32 -> 64x32x32 -> 64x32x32 -> MaxPool -> 64x16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),                     # [B, 64*16*16]
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    #standard forward pass: conv layers -> fc layers
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

num_classes = len(idx_to_name) #number of distinct animal classes
model = AnimalsCNN(num_classes=num_classes).to(device)
print(model)

#count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,} | Trainable: {trainable_params:,}")

#%%
# ------------------------------------------------------------
# 6. Training + evaluation
# ------------------------------------------------------------
#We use `CrossEntropyLoss` as our classification loss and optimize the network parameters with the Adam optimizer. 
#For each batch, we perform a forward pass to obtain the logits, compute the cross-entropy loss against the ground-truth label indices, backpropagate the gradients, and update the weights. 
#We accumulate the batch losses and track how many predictions match the true labels in order to compute the training accuracy.

# After each epoch, we evaluate the model on the test set, and we computes predictions with `argmax` over the output logits, and returns the overall percentage of correctly classified test images. 



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def compute_accuracy(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total

epochs = 5
train_accs, test_accs, losses = [], [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (preds == labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct_train / total_train
    test_acc  = compute_accuracy(testloader)

    losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss={avg_loss:.4f} | "
          f"Train Acc={train_acc:.2f}% | "
          f"Test Acc={test_acc:.2f}%")

#%%
# ------------------------------------------------------------
# 7. Learning curves
# ------------------------------------------------------------
#To analyze the training dynamics, we plot the evolution of the average training loss and the classification accuracy over epochs. 
# The first subplot shows the training loss decreasing as the optimizer minimizes the cross-entropy objective, while the second subplot compares training accuracy to test accuracy for each epoch. 
# ---> CHANGE!!! A healthy training run should exhibit decreasing loss and increasing accuracies; a large gap between training and test performance would indicate overfitting. 
# Here, we can observe how well the CNN learns to classify the 10 animal classes from the Open Images dataset.


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(range(1, epochs+1), losses, marker='o', label='Train Loss')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(range(1, epochs+1), train_accs, marker='o', label='Train Acc')
axes[1].plot(range(1, epochs+1), test_accs, marker='s', label='Test Acc')
axes[1].set_title("Train vs Test Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.suptitle("Animals CNN Learning Curves", fontsize=14)
plt.tight_layout()

results_dir = f".{os.sep}results{os.sep}untrack{os.sep}media{os.sep}"
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "animals_cnn_curves.png"), dpi=300, bbox_inches='tight')
plt.show()

#%% [markdown]
# 7b. Sample battch visualization 
#we visually verify that images and labels are aligned correctly, by taking a mini-batch of normalized images and their integer labels, roughly unnormalizing the images back to `[0, 1]`, and we display a few examples in a row with their decoded class names as titles. 



#%%
# ------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------
models_dir = f".{os.sep}data{os.sep}untrack{os.sep}models{os.sep}"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "cnn_animals10_from_filtered.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")
