# Project: Food Image Analysis for Nutritional Estimation
![image](https://github.com/user-attachments/assets/eca2e5fb-8c5f-483b-a1d9-d4d2344b9e4f)

The goal of this program is to analyze the nutritional value of a meal based on a photo. By leveraging computer vision models, the program can classify food, segment it by type, and estimate nutritional components, including calorie count, carbs, proteins, fats, vitamins, and minerals.

# Features
- V1: Classifying Food from Photo - 
Classifies the food in the image into one of 101 predefined categories.

- V2 (Current): Detecting and Classifying Food Images -
Leverages object detection to identify objects in the photos. All objects that are classified as unknown food types are further identified by type using the food classification model.

- V3 (Future Direction): Portion Approximation & Nutrient Calculation -
Estimates the portion size of each food type in the image and calculates the nutritional contents, including calories, carbs, proteins, fats, vitamins, and minerals.

# Files Overview
- foodimage-classification_training.ipynb:
A Jupyter notebook used to train the computer vision classification model for 101 different food types/classes.

- classify-image-backend.py:
The backend script for an HTML interface that allows users to upload a food image. The program returns a classification label for the food type.
Associated Frontend file: templates/classify-index.html

- object-detection-backend.py:
The backend script for an HTML interface where users can upload an image. The program identifies the major objects in the image using YOLOv5s.

- detect-and-classify-backend.py:
The backend script for an HTML interface where users can upload an image. The program identies the major objects in the image, and further classifies any food object according to the classification model.
Associated Frontend file : templates/detect-and-classify-index.html

- segment-and-classify.ipynb:
A Jupyter notebook used for segmenting food images into different types and returning the classification labels for each segment.

- Checkpoints Folder:
Contains the .pth files that were generated during the training process. These are the saved model states used for inference.


