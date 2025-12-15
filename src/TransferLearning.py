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
test_images_csv  = f'.{os.sep}BigData/test_images.csv'
test_ann_csv     = f'.{os.sep}BigData/test_annotations.csv'
val_images_csv  = f'.{os.sep}BigData/val_images.csv'
val_annotations_csv     = f'.{os.sep}BigData/val_annotations.csv'

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

#Merge validation as well
validation_images = pd.read_csv(val_images_csv)
validation_ann = pd.read_csv(val_annotations_csv)
validation_ann    = validation_ann[validation_ann["LabelName"].isin(label_to_animal.keys())].copy()
validation_ann["ClassName"] = validation_ann["LabelName"].map(label_to_animal)

validation_labels_merged = validation_images.merge(
    validation_ann[["ImageID", "LabelName", "ClassName"]],
    on="ImageID",
    how="inner"
)

# Map class names to numeric variables 
classes = sorted(train_labels_merged["ClassName"].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
num_classes = len(classes)

train_labels_merged["label"] = train_labels_merged["ClassName"].map(class_to_idx)
test_labels_merged["label"]   = test_labels_merged["ClassName"].map(class_to_idx)
validation_labels_merged["label"]  = validation_labels_merged["ClassName"].map(class_to_idx)

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

train_df, test_df = train_test_split(
    train_labels_merged,
    test_size=0.2,             # 20% test, 80% train
    stratify=train_labels_merged['label'],  # keep class proportions
    random_state=42            # for reproducibility
)

train = OpenImagesDataset(train_df, data_transforms["train"])
test   = OpenImagesDataset(test_df, data_transforms["val"])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader  = DataLoader(test, batch_size=32, shuffle=False)

#%%

def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
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

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Train Acc: {acc:.2f}%")

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

# %%
# Now we can train the models

train_model(model, train_loader, optimizer, criterion, device, epochs=3)
#1st epoch was about 13 mins, loss 1.8037, train accuracy 29.24
#2nd epoch done by 17 mins, loss 1.4242, train accuracy 49.52
#3rd epoch at 21 min, loss 1.3022, train acc 53.52

# %%
test_model(model, test_loader, device)

#%%
# Unfreezing the last few layers (prevents overfitting)
for name, param in list(model.named_parameters())[-10:]:
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
train_model(model, train_loader, optimizer, criterion, device, epochs=2)
# Epoch 1, 3.3 min, loss = 1.2081, training accuracy = 56.43%
# Epoch 2, by 6 min, loss = 1.2081, training accuracy = 65.36%
#%%
test_model(model, test_loader, device)

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
# Accuracy; Confusion matrix; Per-class metrics; Train/validation curves
#https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

