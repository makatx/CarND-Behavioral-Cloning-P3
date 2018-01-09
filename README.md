# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_img.jpg "Sample Capture"
[image2]: ./images/center_crop.jpg "Sample Capture, Cropped"
[image3]: ./images/recovery1.jpg "Recovery Image"
[image4]: ./images/recovery2.jpg "Recovery Image"
[image5]: ./images/cnn-architecture.png "CNN Architecture"
[image6]: ./images/receovery_dirt.jpg "Recovery, dirt"
[image7]: ./images/center_img_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network - This file couldn't be uploaded to gitHub as it is 115MB (gitHub's limit is 100MB)
* Writeup in readme.md

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the CNN model from the [Nvidia End-to-End deep learning for self-driving car paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for this project with CNN architecture as shown below:
![image5]

Prior to feeding the images into the network, a 'Lambda' Keras layer is used normalize the pixel data in the given image, after a Cropping2D Keras layer crops the image at the top and bottom, so that the CNN focuses only on the useful parts of the input.
![image1]
![image2]


#### 2. Attempts to reduce overfitting in the model

The model was trained on data collected from the track driven forward and backwards to help it generalize better. Training data was also augumented by flipping to help further.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road so that the model knew how to handle such cases.
Furthermore, additional training data was captured at road areas demarked with dirt road as its limits, since such markings for the road were too few for the model to learn from (twice in the original track).
![image1] ![image7]
![image3] 
![image4]
![image6]


