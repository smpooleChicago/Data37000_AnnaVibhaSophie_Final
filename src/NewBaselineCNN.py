#%%
# ------------------------------------------------------------
# Imports and device
# ------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
# In this section we load the pre-filtered Open Images CSV files that contain
# only our target animal classes. We define a manual dictionary `label_to_animal`
# that maps these Open Images label IDs (e.g. `/m/0bt9lr`) to human-readable
# animal names such as `"Dog"` or `"Cat"`. We first filter the annotation tables
# to keep only rows whose label IDs appear in this dictionary, and then add a
# `ClassName` column with the corresponding animal name.
# Finally, we merge the image lists with the filtered annotations on `ImageID`
# (for both train and test), so each row in `train_labels_merged` and
# `test_labels_merged` contains:
# (1) the image identifier and its metadata,
# and (2) the associated label ID and animal class name.

#%%
# Base directory (BigData folder)
base_dir = "/Users/annasirtori/localDocs/GitHub/Data37000_AnnaVibhaSophie_Final/src/BigData"

train_images_csv = os.path.join(base_dir, "train-images-with-labels-with-rotation_animalsProject.csv")
train_ann_csv    = os.path.join(base_dir, "train-annotations-human-imagelabels_animalsProject.csv")
test_images_csv  = os.path.join(base_dir, "test-images-with-rotation_animalsProject.csv")
test_ann_csv     = os.path.join(base_dir, "test-annotations-human-imagelabels_animalsProject.csv")

train_labels = pd.read_csv(train_images_csv)
classes      = pd.read_csv(train_ann_csv)

label_to_animal = {
    "/m/0bt9lr": "Dog",
    "/m/01dws":  "Bear",
    "/m/03k3r":  "Horse",
    "/m/015p6":  "Bird",
    "/m/01yrx":  "Cat",
    "/m/09ld4":  "Frog",
    "/m/0ch_cf": "Fish",
    "/m/078jl":  "Snake",
    "/m/07bgp":  "Sheep",
    "/m/07pbfj": "Fish",
    "/m/08hhz2": "Sheep"
}

# Keep only the labels we care about (our animal classes)
classes = classes[classes["LabelName"].isin(label_to_animal.keys())].copy()
classes["ClassName"] = classes["LabelName"].map(label_to_animal)

# Merge train image list with annotations
train_labels_merged = train_labels.merge(
    classes[["ImageID", "LabelName", "ClassName"]],
    on="ImageID",
    how="inner"
)

print("Train labels merged shape:", train_labels_merged.shape)
print(train_labels_merged.head())

# Do the same for the test split
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
## 2. Map class names to integer indices + train/val split
# ------------------------------------------------------------
# As Neural networks cannot work directly with string labels, we convert our
# animal classes into integer indices. We first collect the set of distinct
# animal names that appear in `ClassName`, sort them, and assign each one a
# class index via the dictionary `class_to_idx` (e.g. `"Bear" -> 0`, `"Bird" -> 1`, etc.).
# The list `idx_to_name` stores the reverse mapping from index back to class name.
#
# We then add a numeric `'label'` column to the train and test DataFrames using
# this mapping. Finally, we split the merged training table into an internal
# training set (80%) and validation set (20%) using `train_test_split` with
# stratification on the numeric label, so class proportions are preserved in
# both splits.

#%%
# Map class names to integer indices
classes_names = sorted(train_labels_merged["ClassName"].unique())
class_to_idx = {name: i for i, name in enumerate(classes_names)}
idx_to_name  = classes_names
num_classes  = len(classes_names)

# Add numeric 'label' column
train_labels_merged["label"] = train_labels_merged["ClassName"].map(class_to_idx)
test_labels_merged["label"]  = test_labels_merged["ClassName"].map(class_to_idx)

print("Classes:", classes_names)
print("Example train row:\n", train_labels_merged.head(2))

# Split merged train table into train / validation
train_df, val_df = train_test_split(
    train_labels_merged,
    test_size=0.2,
    stratify=train_labels_merged["label"],  # keep class proportions per animal
    random_state=42
)

print(f"\nTrain split size: {len(train_df)} | Val split size: {len(val_df)}")

#%% [markdown]
# ------------------------------------------------------------
## 3. Custom Dataset and DataLoader construction (URL-based)
# ------------------------------------------------------------
# We define a custom PyTorch `Dataset` class `OpenImagesAnimalsDataset` that wraps
# the merged train/val/test DataFrames. Each row contains:
#   - an `OriginalURL` pointing to the image on the web
#   - a numeric `label` column (class index) derived from `ClassName`.
# In `__getitem__` we download the image from the URL, convert it to RGB,
# and pass it through a torchvision transform pipeline to resize, augment,
# convert to a tensor, and normalize its pixel values.
# The dataset therefore returns pairs `(image_tensor, label_idx)`, where
# `image_tensor` has shape `[3, 64, 64]` (after resizing) and `label_idx`
# is an integer in `[0, num_classes)`. These datasets are then wrapped in
# DataLoaders that handle batching and shuffling, producing mini-batches
# ready to be fed into the CNN.

#%%
def load_image_from_url(url):
    """
    Download an image from the given URL and return a PIL.Image in RGB.
    If anything goes wrong (timeout, HTTP error, etc.), return None.
    """
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception:
        return None


class OpenImagesAnimalsDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: DataFrame with at least the columns:
            - 'OriginalURL' : URL of the image
            - 'label'       : integer class index
        transform: torchvision transforms to apply to the PIL image
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        url   = row["OriginalURL"]     # image URL from the CSV
        label = int(row["label"])      # numeric label from ClassName → index

        img = load_image_from_url(url)
        if img is None:
            # If download failed, try the next sample as a simple fallback
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label

#%% [markdown]
# ------------------------------------------------------------
## 4. Transforms and DataLoaders (train / val / test)
# ------------------------------------------------------------
# All images are resized to 64x64 pixels, converted to tensors, and normalized
# channel-wise to have approximate range in [-1, 1]. For training, we apply
# random horizontal flips, small rotations, and color jitter to improve
# robustness and reduce overfitting. For validation and test, we only resize
# and normalize (no augmentation).

#%%
IMG_SIZE = 64  # every image resized to 64x64

# Train images: resize + augmentation + normalize
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Val/Test images: resize + normalize (no augmentation)
transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Datasets from URLs
train_dataset = OpenImagesAnimalsDataset(train_df,          transform=transform_train)
val_dataset   = OpenImagesAnimalsDataset(val_df,            transform=transform_eval)
test_dataset  = OpenImagesAnimalsDataset(test_labels_merged, transform=transform_eval)

# DataLoaders
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
valloader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
testloader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain samples: {len(train_dataset)} | "
      f"Val samples: {len(val_dataset)} | "
      f"Test samples: {len(test_dataset)}")

# sanity check
imgs, labels = next(iter(trainloader))
print("Batch images shape:", imgs.shape)   # [B, 3, 64, 64]
print("Batch labels shape:", labels.shape) # [B]

#%% [markdown]
# ------------------------------------------------------------
## 4b. Sample visualization
# ------------------------------------------------------------
# We visually verify that images and labels are aligned correctly by taking a
# mini-batch of normalized images and their integer labels, roughly unnormalizing
# the images back to [0, 1], and displaying a few examples in a row with their
# decoded class names as titles.

#%%
def show_sample_batch(images, labels, idx_to_name, num_samples=8):
    """
    Show a few images from a batch with their decoded class names.
    images: tensor [B, 3, H, W] (normalized)
    labels: tensor [B]
    """
    # unnormalize: x * std + mean; here mean=std=0.5 for all channels
    images = images * 0.5 + 0.5  # back to [0,1] approx

    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2))
    if num_samples == 1:
        axes = [axes]

    # Show num_samples images from the batch
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

#%% [markdown]
# ------------------------------------------------------------
## 5. CNN model
# ------------------------------------------------------------
# We build a simple 2-block convolutional neural network for classifying the
# animal images. Each block applies two 3x3 convolutions with ReLU activations,
# followed by a 2x2 max pooling layer that halves the spatial resolution.
# Starting from a 3x64x64 RGB image, the first block produces 32x32x32 feature
# maps, and the second block produces 64x16x16 feature maps. A Dropout(0.25)
# layer is added after the convolutional blocks to reduce overfitting.
#
# The fully connected classifier head flattens the 64x16x16 feature maps into
# a 16,384-dimensional vector per image, reduces this to a 256-dimensional
# embedding with a linear layer and ReLU, applies Dropout(0.5), and finally
# maps to `num_classes` output logits. During training, these logits are fed
# into CrossEntropyLoss, which compares the resulting probabilities to the
# true integer label index.

#%%
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

    # standard forward pass: conv layers -> fc layers
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

num_classes = len(idx_to_name)  # number of distinct animal classes
model = AnimalsCNN(num_classes=num_classes).to(device)
print(model)

# count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,} | Trainable: {trainable_params:,}")

#%% [markdown]
# ------------------------------------------------------------
## 6. Training + evaluation
# ------------------------------------------------------------
# We use CrossEntropyLoss as our classification loss and optimize the network
# parameters with the Adam optimizer. For each batch, we perform a forward pass
# to obtain the logits, compute the cross-entropy loss against the ground-truth
# label indices, backpropagate the gradients, and update the weights. We
# accumulate the batch losses and track how many predictions match the true
# labels to compute the training accuracy.
#
# After each epoch, we evaluate the model on the validation set, computing
# the overall percentage of correctly classified validation images. The
# separate filtered test split is kept untouched until the end and used
# once to report the final test accuracy.

#%%
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
train_accs, val_accs, losses = [], [], []

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
    val_acc   = compute_accuracy(valloader)

    losses.append(avg_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss={avg_loss:.4f} | "
          f"Train Acc={train_acc:.2f}% | "
          f"Val Acc={val_acc:.2f}%")

# Final test accuracy on the held-out test set
final_test_acc = compute_accuracy(testloader)
print(f"\nFinal test accuracy on held-out test set: {final_test_acc:.2f}%")

#%% [markdown]
# ------------------------------------------------------------
## 7. Learning curves
# ------------------------------------------------------------
# To analyze the training dynamics, we plot the evolution of the average
# training loss and the classification accuracy over epochs. The first subplot
# shows the training loss decreasing as the optimizer minimizes the cross-entropy
# objective, while the second subplot compares training accuracy to validation
# accuracy for each epoch. A healthy training run should exhibit decreasing loss
# and increasing accuracies; a large gap between training and validation
# performance would indicate overfitting. Here, we can observe how well the CNN
# learns to classify the animal classes from the Open Images dataset.

#%%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(range(1, epochs+1), losses, marker='o', label='Train Loss')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(range(1, epochs+1), train_accs, marker='o', label='Train Acc')
axes[1].plot(range(1, epochs+1), val_accs,   marker='^', label='Val Acc')
axes[1].set_title("Train vs Validation Accuracy")
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
## 7b. Sample batch visualization (after training)
# We can re-use the `show_sample_batch` function to visualize a batch of
# training or test images and their predicted or true labels, to gain more
# intuition about the model's behavior.

#%% [markdown]
# ------------------------------------------------------------
## 8. Save model
# ------------------------------------------------------------
# We save only the model weights (`state_dict`) to a .pth file so that the
# trained CNN can be reloaded later for further evaluation or inference.

#%%
models_dir = f".{os.sep}data{os.sep}untrack{os.sep}models{os.sep}"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "cnn_animals10_from_filtered.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")