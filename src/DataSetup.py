#%%[markdown]
# Some of the image and annotations files were too large to share through GitHub,
# therefore, from the class descriptions csv, we chose 10 animal classes and
# re-saved the files with only those classes. 
#
# For our project, we'll be using classes: Dog, Bird, Horse, Cat, Fish, Frog, Bear, Snake, Sheep, Sea Lion.
# We also made sure that these animals classes had at least 100 images each. 
#
# YOU NEED THE ORIGINAL FILES TO RUN THIS

# %%

import pandas as pd

# Path to your downloaded class-descriptions file
csv_path = "class-descriptions.csv"

classes = pd.read_csv("class-descriptions.csv", header=None, names=["MID", "ClassName"])

labels = pd.read_csv("train-annotations-human-imagelabels.csv")

# %%
# Get animal classes that have at least 100 images each

counts = labels.groupby("LabelName")["ImageID"].nunique().reset_index()
counts.columns = ["MID", "ImageCount"]

counts = counts.merge(classes, on="MID", how="left")
counts = counts.sort_values("ImageCount", ascending=False)

counts = counts[counts["ImageCount"] >=100]

#Get classes that contains animal keywords
animal_keywords = [
    "animal", "cat", "dog", "fish", "bird", "bear", "horse", "cow", "sheep",
    "goat", "pig", "chicken", "duck", "lion", "tiger", "elephant", "monkey",
    "giraffe", "zebra", "insect", "bug", "spider", "reptile", "snake", "frog"
]

isAnimal = counts["ClassName"].str.contains("|".join(animal_keywords), case=False, regex=True)

#Save the names to file all_animal_classes.csv
#counts[isAnimal]["ClassName"].to_csv("all_animal_classes.csv", index=False)

# %%
# Saves test, train, and validation files with only the classes we want for our project

animalsProject = ["Dog", "Bird", "Horse", "Cat", "Bear","Sheep", "Cattle"]
#animalsProject = ['Dog']
target = classes[classes["ClassName"].isin(animalsProject)]
target_MID = target["MID"].tolist()

#train-annotations-human-imagelabels.csv but only with classes that fit animalsProject
filtered = labels[labels["LabelName"].isin(target_MID)]

#saves filtered (same file but with _animalsProject at the end)
#filtered.to_csv("train-annotations-human-imagelabels_animalsProject.csv", index=False)

#From the filtered MID, filters test annotations
test_annotations = pd.read_csv("test-annotations-human-imagelabels.csv")
filtered_test_annotations = test_annotations[test_annotations["LabelName"].isin(target_MID)]
#filtered_test_annotations.to_csv("test-annotations-human-imagelabels_animalsProject.csv", index=False)

#From the filtered test annotations, filters test images
test_images = pd.read_csv("test-images-with-rotation.csv")
filtered_ID = filtered["ImageID"].tolist()
test_images_filtered = test_images[test_images["ImageID"].isin(filtered_test_annotations["ImageID"].tolist())]
#test_images_filtered.to_csv("test-images-with-rotation_animalsProject.csv", index=False)

#From the filtered MID, filters train annotations
train_annotations = pd.read_csv("train-annotations-human-imagelabels.csv")
filtered_train_annotations = train_annotations[train_annotations["LabelName"].isin(target_MID)]
#filtered_train_annotations.to_csv("train-annotations-human-imagelabels_animalsProject.csv", index=False)

#From the filtered train annotations, filters train images
train_images = pd.read_csv("train-images-with-labels-with-rotation.csv")
filtered_ID = filtered["ImageID"].tolist()
train_images_filtered = train_images[train_images["ImageID"].isin(filtered_train_annotations["ImageID"].tolist())]
#train_images_filtered.to_csv("train-images-with-labels-with-rotation_animalsProject.csv", index=False)

#From filtered MID, filters validation annotations
validation_annotations = pd.read_csv("validation-annotations-human-imagelabels.csv")
filtered_validation_annotations = validation_annotations[validation_annotations["LabelName"].isin(target_MID)]
#filtered_validation_annotations.to_csv("validation-annotations-human-imagelabels_animalsProject.csv", index=False)

#From the filtered validation annotations, filters validation images
validation_images = pd.read_csv("validation-images-with-rotation.csv")
filtered_ID = filtered["ImageID"].tolist()
validation_images_filtered = validation_images[validation_images["ImageID"].isin(filtered_validation_annotations["ImageID"].tolist())]
#validation_images_filtered.to_csv("validation-images-with-rotation_animalsProject.csv", index=False)
# %%
# Gets 150 valid images for each class

filtered_train_annotations_ran = (
    filtered_train_annotations.groupby("LabelName", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 200), random_state=42))
      .reset_index(drop=True)
)

train_images_filtered_ran = train_images_filtered[train_images_filtered["ImageID"].isin(filtered_train_annotations_ran["ImageID"].tolist())]

import requests

available_rows = []

for _, row in train_images_filtered_ran.iterrows():
    url = row["OriginalURL"]
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            available_rows.append(row)  # keep the entire row
    except:
        pass

train_images = pd.DataFrame(available_rows)
train_annotations = filtered_train_annotations_ran[filtered_train_annotations_ran["ImageID"].isin(train_images["ImageID"].tolist())]
train_annotations = (
    train_annotations.groupby("LabelName", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 150), random_state=42))
      .reset_index(drop=True)
)
train_images = train_images[train_images["ImageID"].isin(train_annotations["ImageID"].tolist())]

train_images.to_csv("train_images.csv", index=False)
train_annotations.to_csv("train_annotations.csv", index=False)

# %%
#Gets 30 valid images for each class

filtered_test_annotations_ran = (
    filtered_test_annotations.groupby("LabelName", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 100), random_state=42))
      .reset_index(drop=True)
)

test_images_filtered_ran = test_images_filtered[test_images_filtered["ImageID"].isin(filtered_test_annotations_ran["ImageID"].tolist())]

import requests

available_rows = []

for _, row in test_images_filtered_ran.iterrows():
    url = row["OriginalURL"]
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            available_rows.append(row)  # keep the entire row
    except:
        pass

test_images = pd.DataFrame(available_rows)
test_annotations = filtered_test_annotations_ran[filtered_test_annotations_ran["ImageID"].isin(test_images["ImageID"].tolist())]
test_annotations = (
    test_annotations.groupby("LabelName", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 30), random_state=42))
      .reset_index(drop=True)
)
test_images = test_images[test_images["ImageID"].isin(test_annotations["ImageID"].tolist())]

#test_images.to_csv("test_images.csv", index=False)
#test_annotations.to_csv("test_annotations.csv", index=False)


# %%
# Gets 30 valid images from each class for validation (which is also 20% of train)

filtered_validation_annotations_ran = (
    filtered_validation_annotations.groupby("LabelName", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 100), random_state=42))
      .reset_index(drop=True)
)

validation_images_filtered_ran = validation_images_filtered[validation_images_filtered["ImageID"].isin(filtered_validation_annotations_ran["ImageID"].tolist())]

import requests

available_rows = []

for _, row in validation_images_filtered_ran.iterrows():
    url = row["OriginalURL"]
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            available_rows.append(row)  # keep the entire row
    except:
        pass

val_images = pd.DataFrame(available_rows)
val_annotations = filtered_validation_annotations_ran[filtered_validation_annotations_ran["ImageID"].isin(val_images["ImageID"].tolist())]
val_annotations = (
    val_annotations.groupby("LabelName", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 30), random_state=42))
      .reset_index(drop=True)
)
val_images = val_images[val_images["ImageID"].isin(val_annotations["ImageID"].tolist())]

#val_images.to_csv("val_images.csv", index=False)
#val_annotations.to_csv("val_annotations.csv", index=False)
# %%
