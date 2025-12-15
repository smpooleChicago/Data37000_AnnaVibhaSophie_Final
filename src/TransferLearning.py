#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import requests
from PIL import Image
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
from sklearn.metrics import confusion_matrix



#%% [markdown]
# # Transfer Learning
# To improve upon the baseline CNN image classification model, we will be 
# using Transfer Learning. We will be using an already trained model (trained 
# on ImageNet), and adapt it to our current image dataset. 
#%%
# ### Setup 
# To start, we will be using the same subset as the CNN baseline model. 

device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
device = torch.device(device)

train_images_csv = f'.{os.sep}BigData/train_images.csv'
train_ann_csv    = f'.{os.sep}BigData/train_annotations.csv'

train_labels = pd.read_csv(train_images_csv)   # ImageID + rotation + etc.
classes      = pd.read_csv(train_ann_csv)      # ImageID + LabelName + Confidence (for filtered animals)

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

# Map class names to numeric variables 
classes = sorted(train_labels_merged["ClassName"].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
num_classes = len(classes)

train_labels_merged["label"] = train_labels_merged["ClassName"].map(class_to_idx)

#%%
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()  # check HTTP errors
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Skipping URL {url}: {e}")
        return None


class OpenImagesDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        start_idx = idx
        while True:
            row = self.df.iloc[idx]
            img = load_image_from_url(row["OriginalURL"])
            if img is not None:
                if self.transform:
                    img = self.transform(img)
                return img, row["label"]

            idx = (idx + 1) % len(self.df)
            if idx == start_idx:
                return None
    
from sklearn.model_selection import train_test_split

train_df, test_and_val = train_test_split(
    train_labels_merged,
    test_size=0.3, # 70% train
    stratify=train_labels_merged['label'], 
    random_state=42 
)

test_df, val_df = train_test_split(
    test_and_val,
    test_size=0.5, #15% test, 15% validation
    stratify=test_and_val['label'],
    random_state=42
)

train = OpenImagesDataset(train_df, data_transforms["train"])
test   = OpenImagesDataset(test_df, data_transforms["val"])
val = OpenImagesDataset(val_df, data_transforms["val"])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader  = DataLoader(test, batch_size=32, shuffle=False)
val_loader = DataLoader(val, batch_size=32, shuffle=False)

#%%

def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss.append(running_loss / total)
        train_acc.append(correct/total)
        acc = 100 * correct / total

        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss.append(val_running_loss / val_total)
        val_acc.append(val_correct / val_total )
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Train Acc: {acc:.2f}%")
    return train_loss, train_acc, val_loss, val_acc

def test_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# %%
# From TorchVision, we import the ResNet 18 model (which is a fully pretrained model). 
# 
# The 'out_features' of the nn.Linear function is set to the number of classes we have, 
# which is 9. 

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(set(train_labels_merged["ClassName"])))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

#%% [markdown]
# Now we can train the models
# 
# * Epoch 1: loss = 1.8721, train accuracy = 24.9%
# * Epoch 2: loss = 1.5139, train accuracy = 43.54%
# * Epoch 3: loss = 1.3147, train accuracy = 54.01%

train_loss, train_acc, val_loss, val_acc = train_model(model, train_loader, optimizer, criterion, device, epochs=3)


# %%
test_model(model, test_loader, device)

#%% [markdown]
# Unfreezing the last few layers (prevents overfitting)
#
# * Epoch 1: loss = 1.2317, training accuracy = 53.74%
# * Epoch 2: loss = 1.0012, training accuracy = 65.31%

for name, param in list(model.named_parameters())[-10:]:
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
train_loss_unfreeze, train_acc_unfreeze, val_loss_unfreeze, val_acc_unfreeze = train_model(model, train_loader, optimizer, criterion, device, epochs=2)

#%%
test_model(model, test_loader, device)

#%% [markdown]
# # Model Evaulation
#
# Rows represents the true labels and the columns are the predicted labels
# 

model.eval()

y_true = []
y_pred = []

with torch.no_grad():  # no gradients needed during evaluation
    for images, labels in test_loader:
        images = images.to(device)  # move to GPU if available
        labels = labels.to(device)

        outputs = model(images)           # forward pass
        _, preds = torch.max(outputs, 1)  # get predicted class index

        y_true.extend(preds.cpu().numpy())
        y_pred.extend(labels.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print(cm)
#%% [markdown]
# ## Per-Class Metrics
# The class with the highest precision is Bird and the class with the lowest precision is Sheep. 
# The support shows that there were only 2 available images used for Sheep. This could be due to the 429 
# error (too many requests from Open Images) or links being outdated. 
#
ClassNames = ["Dog", "Bird", "Horse", "Cat", "Bear", "Sheep", "Cattle"]
from sklearn.metrics import classification_report
classReport = classification_report(y_true, y_pred, target_names=ClassNames)
print(classReport)
#%% [markdown]
# ## Training vs Validation Curve
# 
# The first plot shows the loss curve (i.e. Training Loss vs epochs and Validation Loss vs epochs) and
# the second plot shows the accuracy urve (i.e. Training accuracy vs epoch and Validation accuracy vs epoch). 
#
# As the number of epochs increase, generally the loss decreases and accuracy increases. However we can see that 
# Train loss curve is very low for all the epochs and at the end the validation loss curve goes upwards. 
# This could be signs of overfitting. 
#
# The learning curves are generally going upwards (a good sign that the model is learning). 
# However the loss curve suggests that the learning isn't stable. 

import matplotlib.pyplot as plt
epochs = range(1, 6)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss+train_loss_unfreeze, label='Train Loss')
plt.plot(epochs, val_loss + val_loss_unfreeze, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy curve
plt.subplot(1,2,2)
plt.plot(epochs, train_acc + train_acc_unfreeze, label='Train Acc')
plt.plot(epochs, val_acc + val_acc_unfreeze, label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.show()


#%%
sample, _ = next(iter(test_loader))
x = sample[0].unsqueeze(0).to(device)
with torch.no_grad():
    features = model.conv1(x).cpu().squeeze()

fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(features[i], cmap='gray')
    ax.axis('off')
plt.suptitle("First Conv Layer Feature Maps")
plt.show()

# %%
# CNN

transform = transforms.Compose([
    transforms.Resize((128, 128)),   # ⭐ HERE
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train = OpenImagesDataset(train_df, transform)
test   = OpenImagesDataset(test_df, transform)
val = OpenImagesDataset(val_df, transform)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader  = DataLoader(test, batch_size=32, shuffle=False)
val_loader = DataLoader(val, batch_size=32, shuffle=False)

# %%
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

            nn.AdaptiveAvgPool2d((16, 16)),

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
    

num_classes = len(set(train_labels_merged)) #number of distinct animal classes
modelCNN = AnimalsCNN(num_classes=num_classes).to(device)
print(modelCNN)

#count total and trainable parameters
total_params = sum(p.numel() for p in modelCNN.parameters())
trainable_params = sum(p.numel() for p in modelCNN.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,} | Trainable: {trainable_params:,}")

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelCNN.parameters(), lr=1e-3)

def compute_accuracy(loader):
    modelCNN.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = modelCNN(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total

epochs = 5
train_accs, val_accs, test_accs, losses = [], [], [], []

for epoch in range(epochs):
    modelCNN.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = modelCNN(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (preds == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct_train / total_train
    val_acc   = compute_accuracy(val_loader)
    #test_acc  = compute_accuracy(testloader)
    losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(val_acc)  # you can rename test_accs -> val_accs if you prefer

    print(f"Epoch {epoch+1}/{epochs} | "
      f"Loss={avg_loss:.4f} | "
      f"Train Acc={train_acc:.2f}% | "
      f"Val Acc={val_acc:.2f}%")
    
# * Epoch 1/5, loss = 2.1854, train acc = 13.33%, val acc = 14.56%
# * Epoch 2/5, loss=2.0052, train Acc=16.05%, val acc=22.15%
# * Epoch 3/5, loss=1.9327, train Acc=21.77%, val acc=24.68%
# * Epoch 4/5, loss=1.8845, train acc=25.71%, val acc = 22.15%
# * Epoch 5/5, loss=1.8524, train acc=26.94%, val acc = 24.05%
# %%

test_model(modelCNN, test_loader, device)
# 27.39%

# %%
modelCNN.eval()

y_true = []
y_pred = []

with torch.no_grad():  # no gradients needed during evaluation
    for images, labels in test_loader:
        images = images.to(device)  # move to GPU if available
        labels = labels.to(device)

        outputs = modelCNN(images)           # forward pass
        _, preds = torch.max(outputs, 1)  # get predicted class index

        y_true.extend(preds.cpu().numpy())
        y_pred.extend(labels.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print(cm)
#%% [markdown]

ClassNames = ["Dog", "Bird", "Horse", "Cat", "Bear", "Sheep", "Cattle"]
from sklearn.metrics import classification_report
classReport = classification_report(y_true, y_pred, target_names=ClassNames)
print(classReport)
# %%
