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

animalsProject = ["Dog", "Bird", "Horse", "Cat", "Fish", "Frog", "Bear", "Snake", "Sheep", "Sea Lion"]
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
# %%
