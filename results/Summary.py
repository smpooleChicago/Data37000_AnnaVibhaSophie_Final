#%%[markdown]
# # Summary
# In this project, we wanted to use image classification on animal images. 
# We used the images from OpenImages V4. We chose images from 7 different animal classes:
# Dog, Bird, Horse, Cat, Bear, Sheep, and Cattle

# For our baseline model, we used CNN and attempted to improve the model using Transfer Learning. 
# We chose Transfer Learning specifically since uses pre-trained CNN's on images, such as from ImageNet, 
# and has reduced training time since it only needs to fine-tune for the final layers of the neural network. 
#
# ## Baseline Model
# From the Baseline model, there were 4,261,927 total parameters. 
#
# From the training, 
# * Epoch 1/5, loss = 2.1854, train acc = 13.33%, val acc = 14.56%
# * Epoch 2/5, loss=2.0052, train Acc=16.05%, val acc=22.15%
# * Epoch 3/5, loss=1.9327, train Acc=21.77%, val acc=24.68%
# * Epoch 4/5, loss=1.8845, train acc=25.71%, val acc = 22.15%
# * Epoch 5/5, loss=1.8524, train acc=26.94%, val acc = 24.05%
#
# The test model's accuracy was 27.39%. 
# 
#
# ## Transfer Learning
# 
# From one of the times running the Transfer Learning Model,
# * Epoch 1: loss = 1.8721, train accuracy = 24.9%
# * Epoch 2: loss = 1.5139, train accuracy = 43.54%
# * Epoch 3: loss = 1.3147, train accuracy = 54.01%
#
# We can see from the trend that with each epoch, the loss decreased as the train accuracy increased. 
# And this was the same trend when running other times. 
#
# The Test accuracy was 47.13%. 
# 
# The data went through the training again, but with unfreezing on the last few layers. 
# * Epoch 1: loss = 1.2317, training accuracy = 53.74%
# * Epoch 2: loss = 1.0012, training accuracy = 65.31%
# 
# The test accuracy this time was 52.23%, which is only 5% higher than the initial training. 
#
# # Comparison
# Both models went through 5 epochs, with the training accuracies continuously increasing 
# and the loss continuously decreasing. 
# Both had the same pattern where at the 4th epoch, the validation accuracy decreased instead 
# of increased. This suggests that this data had an unstable learning rate. 
# Comparing the training accuracies and the test accuracies, the accuracies were significantly 
# higher for the Transfer Learning model. This shows how the Transfer Learning *was* an improved 
# model like we intended to make. 
#
# The classification report that bird had the highest precision (for both CNN and Transfer Learning),
# and sheep had the lowest precision for both models. However, looking at the support numbers for the models,
# the precision and recall are generally higher when there are more images for that class. This shows that
# with more data, the model tends to be more accurate. 
# 
# 
# # Discussion and Interpretation
# The hardest roadblock we faced was the 429 error where 
# we were only able to request from Open Images a certain amount of times. 
# We were able to move around this roadblock however by splitting the training
# set into new test and train subsets. This still allowed for a total of 1080 images to be used. 
#
# Image classification already has many real world applications. In the Tesla, it shows 
# whether the object beside you is a vehicle (bus, truck, car) or a person. In the iPhone, 
# the photo app can not only identify whether the image contains a dog, but also classify 
# that breed of dog (though with less accuracy). 
#
# However there are negative sides of Image Classification as well. One example is in the 
# medical field. If Image Classification is used to spot a physical illness or problem, it's 
# accuracy is highly important. A wrong classification can lead to a wrong diagnosis. 
# That's why we believe that in important fields, Image Classification can be used to aid, 
# but not fully decide. 
#
# The Image Classification model also depends on the images used to train the model. 
# There could be an imbalance if one class had significantly more images than another class.
# In our models, we tried to eliminate this problem by choosing the same number of images for each class. 
# However, we picked and chose the classes based on how many images there were. This 
# imbalance would be more worrysome if we were using all the animal classes. 
# 
# If the images were miss-labeled on the training images, that could also lead to a problem 
# (i.e. training a model on images with the wrong name). OpenImages had both human labeled 
# and machine labeled images, and we chose human labeled in hopes they would be more accurate 
# for our project. 
#
# With more time (and computing power), we could use more images to train the model. 
# This should lead to a higher accuracy since there's a larger variety of pictures to learn 
# from for each animal. We could also add more hyperparameter tuning to optimize the model's performance. 
# %%
# ## Credits
# 
# With the help of ChatGPT, we were able to load the image dataset. 
# We did not use any AI to fully create the model or for any brainstorming, evaluation, analyzing, or critical thinking. 
