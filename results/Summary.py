#%%[markdown]
# # Summary
# In this project, we wanted to use image classification on animal images. 
# We used the images from OpenImages V4. We chose images from 7 different animal classes:
# Dog, Bird, Horse, Cat, Bear, Sheep, and Cattle

# For our baseline model, we used CNN and attempted to improve the model using Transfer Learning. 
# We chose Transfer Learning specifically since uses pre-trained CNN's on images, such as from ImageNet, 
# and has reduced training time since it only needs to fine-tune for the final layers of the neural network. 
#
# The baseline model is in the file BaselineCNN.py and the Transfer Learning model 
# is in the file TransferLearning.py (both in the src file).
#
# ## Baseline Model
# From the Baseline model, there were 4,261,927 total parameters. 
#
# ## Transfer Learning
# 
# From one of the times running the Transfer Learning Model,
# * Epoch 1: loss = 1.8037, train accuracy = 29.24%
# * Epoch 2: loss = 1.4242, train accuracy = 49.52%
# * Epoch 3: loss = 1.3022, train accuracy = 53.52%
#
# We can see from the trend that with each epoch, the loss decreased as the train accuracy increased. 
# And this was the same trend when running other times. 
#
# The Test accuracy was 51.43%, so only a little bit above half the training set. 
# 
# The data went through the training again, but with unfreezing on the last few layers. 
# * Epoch 1: loss = 1.2081, training accuracy = 56.43%
# * Epoch 2: loss = 1.2081, training accuracy = 65.36%
# 
# The test accuracy this time was 55.24%, which is only 4% higher than the initial training. 
#
# 
# # Discussion and Interpretation
# The hardest roadblock we faced was the 429 error where 
# we were only able to request from Open Images a certain amount of times. 
# We were able to move around this roadblock however by splitting the training
# set into new test and train subsets. This still allowed for a total of 1080 images to be used. 
#
# Image classification is useful for classifing animals; one of the places where it's used 
# is on the iPhone image app where it can tell what kind of animal the image is showing. 
# It even gets more specific than our models were it can categorize some breeds 
# %%
