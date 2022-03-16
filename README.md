# MLiS-II-Project-Programming-an-Autonomous-Driving-Car
# Table of Contents
1. [Overview](#overview)
2. [Challenge Details](#challenge-details)
    1. [Training data](#training-data)
    2. [Challanges](#challanges)
    3. [Equipment](#equipment)
    4. [Deploying Models to the Car](#deploying-models-to-the-car)
    5. [Training Machines](#training-machines)
    6. [Tracks](#tracks)
    7. [Driving Scenarios](#driving-scenarios)
3. [Assessment and Important Dates](#assessment-and-important-dates)
4. [Teams](#teams)
5. [Resources to build the model](#resources-to-build-the-model)
6. [Tips](#tips)
7. [Report Marking CriteriaFile](#report-marking-criteria)

# Overview
**_In this project the task is to train a deep learning algorithm to autonomously navigate a real car around a realistic test circuit, and make the appropriate manoeuvres where necessary. At the end of the project, you are expected to give a presentation and write a report about what you have done. Your model will be tested on the track and will compete against the models of your peers._**

# Challenge Details
- Work in pairs
- Develop a deep learning model
- Input = Image from a camera on the car
- Predictions = Appropriate speed and steering angle

## Training data
- [Dataset](https://www.kaggle.com/c/machine-learning-in-science-2022/data) is hosted on Kaggle
- 13.8k images
- Spped & Steering angle is also available in the dataset
- We are free to generate our own dataset

## Challanges
1. This will allow you to automate the process of model submission, and obtain an indication of performance (using a small set of test data), before we evaluate them on the final, unseen data.
2. Kaggle competition is hosted [here](https://www.kaggle.com/c/machine-learning-in-science-2022).
3. Create a Kaggle account (if you do not have one) and form a team with your project partner.
4. A live challenge, where your pre-trained model will be deployed to the car and tested on real circuits. This will be performed in person.

## Equipment
- The main body of the car is the SunFounder [PiCar-V kit V2](https://www.sunfounder.com/products/smart-video-car) and is equipped with a Raspberry Pi (RPi)
- **_TensorFlow v2.4_** is installed on the car,
- The car has an optional Coral Edge TPU, which is a custom device to run forward-pass operations for edge computing.
- Note that it isn’t necessary to convert your model to TensorFlow Lite.

## Deploying Models to the Car
A standardised skeleton code will be provided to you that you should integrate your pre-trained model with, which we will then install on the car prior to the live testing.

## Training Machines
- We can use Google Colab or our own local machine to train the model
- We will also have access to MLiS1 or MLiS2 machines (each with w) to perform training. These are accessible by ssh’ing into the machine, by typing 
    - ssh username@mlis1.nottingham.ac.uk or 
    - ssh username@mlis2.nottingham.ac.uk, where username is your University username.
- In order to install custom packages on your machine, you will need to set up a conda environment. To install conda, type the following command 
    - bash /shared/Anaconda3-2019.10-Linux-x86 64.sh
    - Once installed, you will need to add a start up script
    - echo . ∼/.bashrc >> .profile
    - Lastly to create your conda environment use
    - conda create --name my env python=3.6
## Tracks
![](Images/Live_Testing_Track.png)
</br>
Figure 1: 
- (Top) T-junction tracks 
- (Middle) Oval Track 
- (Bottom) Figure-of-eight track.

## Driving Scenarios
**Important**: we only use UK driving rules, i.e. driving on the left-hand side. The training data was based on the following driving scenarios:
1. Keeping in lane driving along the straight section of the T-junction track.
2. As (1), but stopping if a pedestrian is in the road.
3. As (1), but driving as normal if pedestrians or other objects are on the side of (but not in) the road.
4. Driving around the oval track in both directions.
5. As (4), but stopping if a pedestrian is in the road.
6. As (4), but driving as normal if pedestrians or other objects are on the side of (but not in) the road.
7. Performing a turn at the T-junction, in response to a traffic sign (either left or right).
8. Driving around the figure-of-eight track in both directions, continuing straight at the intersection. We will not consider objects in or at the side of the road for this scenario.
9. Stopping at a red traffic light and continuing at a green traffic light. We will only consider these scenarios in the live testing.

# Assessment and Important Dates
- **Kaggle and code submission**: 1.30pm on Friday 6th May 2022
- **Live testing**: Wednesday 11th May 2021
    - live testing performance will contribute 20% towards your total mark
- **In-person Presentations**: 16th and 17th May 2022
    - The presentations will last for 25 minutes
    - speak for approximately 10 minutes each, with 5 minutes for questions
    - You will be penalized if you over- run significantly
    - The presentation will contribute 40% towards your total mark
- **Report**: Friday May 20th 2022
    - This is to be completed individually and will contribute 40% towards your total mark

# Teams
Please come up with a suitable team name and use this on Kaggle
</br>
Baride & Talukdar
</br>
Team name is **_Alpha_**

# Resources to build the model
1. [DeepPiCar — Part 1: How to Build a Deep Learning, Self Driving Robotic Car on a Shoestring Budget](https://towardsdatascience.com/deeppicar-part-1-102e03c83f2c)
2. [TensorFlow models on the Edge TPU](https://coral.ai/docs/edgetpu/models-intro/#compatibility-overview)

# Tips
1. Make good use of your Kaggle submissions to test different things. You get 1 per day – don’t let them go to waste! The only submission that matters is your final one earlier ones have no bearing on your mark, so don’t worry if the test score isn’t great. The best performing groups last year all made many submissions.
2. Get started early and make a plan.
3. Some initial data exploration on the training data will help you understand it better.
4. More tips to come in the introductory session, including lessons from last year!

# Report Marking Criteria
[report-guidelines.pdf](Resources/report-guidelines.pdf)
</br>
[presentation-guidelines.pdf](/Resources/presentation-guidelines.pdf)




