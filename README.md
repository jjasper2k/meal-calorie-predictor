# Project: Food Image Analysis for Nutritional Estimation
![image](https://github.com/user-attachments/assets/eca2e5fb-8c5f-483b-a1d9-d4d2344b9e4f)

In recent decades, global eating habits have taken a troubling trajectory. Increased consumption of processed and calorie-dense foods has greatly contributed to rising rates of obesity, diabetes, and heart disease. Despite public awareness campaigns about the importance of nutrition, many individuals face challenges not only in their ability to make informed dietary decisions, but also in their ability to understand the impact of their daily decisions on their health. Empowering individuals to understand and improve their eating habits can play a critical role in addressing widespread health issues stemming from lack of proper nutrition and fostering a healthier society.

The goal of this program is to analyze the nutritional value of a meal based on a photo. By leveraging advanced computer vision models, the program identifies various foods in an image and classifies them by food type. The future objective is to estimate key nutritional components of the food in the image, such as calorie count, carbohydrates, proteins, fats, vitamins, and minerals. With access to detailed dietary information, users will gain greater control over their eating habits and personal health, enabling more informed decisions that can contribute to improved overall well-being. In the future, aggregating user data over long time periods could allow users to understand the long term impact of their eating habits on their health. Moreover, user data could be used to tailor user-specific dietary plans, granting increased access to low-cost nutritional counseling. This program has the potential to bridge the gap between technology and nutrition, offering a user-friendly solution to help combat diet-related health issues.


# Features
- V1: Classifying Food from Photo - 
Classifies the food in the image into one of 101 predefined categories.

- V2 (Current): Detecting and Classifying Food Images -
Leverages object detection to identify objects in the photos. All objects that are identified as food types are further classified into types using the food classification model.

- V3 (Future Direction): Portion Approximation & Nutrient Calculation -
Estimates the portion size of each food type in the image and calculates the nutritional contents, including calories, carbs, proteins, fats, vitamins, and minerals.

# Files Overview
- foodimage-classification_training.ipynb:
A Jupyter notebook used to train the computer vision classification model for 101 different food types/classes. The food classification model leveraged a ResNet18 model trained using the Food 101 image dataset on Kaggle.

- detect-and-classify-backend.py:
The backend script for an HTML interface where users can upload an image. The program identies the major objects in the image, and further classifies any food object according to the classification model.
Associated Frontend file : templates/detect-and-classify-index.html

- Checkpoints Folder:
Contains the .pth files that were generated during the training process. These are the saved model states used for inference.


