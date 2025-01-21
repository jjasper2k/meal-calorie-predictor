# Project: Food Image Analysis for Nutritional Estimation
The goal of this program is to analyze the nutritional value of a meal based on a photo. By leveraging computer vision models, the program can classify food, segment it by type, and estimate nutritional components, including calorie count, carbs, proteins, fats, vitamins, and minerals.

# Features
- V1 (Current): Classifying Food from Photo - 
Classifies the food in the image into one of 101 predefined categories.

- V2 (Future Direction): Segmenting Food Images -
Segments the image by types of food, then classifies each segment to provide more granular insights.

- V3 (Future Direction): Portion Approximation & Nutrient Calculation -
Estimates the portion size of each food type in the image and calculates the nutritional contents, including calories, carbs, proteins, fats, vitamins, and minerals.

# Files Overview
- foodimage-classification_training.ipynb:
A Jupyter notebook used to train the computer vision classification model for 101 different food types/classes.

- classify-image-backend.py:
The backend script for an HTML interface that allows users to upload a food image. The program returns a classification label for the food type.
Associated Frontend: templates/classify-index.html

- segment-and-classify-backend.py:
The backend script for an HTML interface where users can upload an image. The program segments the image by different food types and classifies each segment.
Associated Frontend: templates/segment-and-classify-index.html

- segment-and-classify.ipynb:
A Jupyter notebook used for segmenting food images into different types and returning the classification labels for each segment.

- Checkpoints Folder:
Contains the .pth files that were generated during the training process. These are the saved model states used for inference.


