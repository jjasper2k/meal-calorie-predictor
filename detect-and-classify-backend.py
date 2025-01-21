from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import base64
import os

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use YOLOv5 small model

# Load trained ResNet-18 model for classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_classes = 101  # Adjust the number of classes if needed
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load model checkpoint
script_dir = os.path.dirname(__file__)
checkpoint_path = os.path.join(script_dir, "checkpoints", "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load class labels from labels.txt
def load_class_labels(labels_file):
    with open(labels_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Path to the labels file
labels_file = os.path.join(script_dir, "food-101", "meta", "labels.txt")
class_labels = load_class_labels(labels_file)

# Define image preprocessing for classification model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to classify the cropped image (e.g., for "bowl" objects)
def classify_image(cropped_image):
    # Ensure the cropped image is in RGB mode
    cropped_image = cropped_image.convert("RGB")
    
    # Resize the cropped image to (224, 224) for the model input
    cropped_image = cropped_image.resize((224, 224))

    # Apply the preprocessing transform
    input_tensor = transform(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    # Map the predicted class index to the class label
    predicted_class_label = class_labels[predicted_class.item()]
    return predicted_class_label
# Function to perform object detection and classify "bowl" objects
def detect_and_classify(file):
    try:
        # Load the image
        image = Image.open(file).convert("RGB")
        img_array = np.array(image)  # Convert to numpy array for YOLO

        # Perform object detection with YOLO
        results = yolo_model(img_array)

        # Extract detection results
        detected_items = results.xyxy[0].numpy()  # Bounding box coordinates and confidence
        labels = results.names  # Class labels

        # Draw bounding boxes and classify "bowl" objects
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for item in detected_items:
            xmin, ymin, xmax, ymax, confidence, class_id = item
            label = labels[int(class_id)]

            if label == "bowl":  # Check for bowl label
                # Crop the detected region (the "bowl" object)
                print("bowl found")
                cropped = image.crop((xmin, ymin, xmax, ymax))
                
                # Classify the cropped region
                classification_label = classify_image(cropped)
                print("image classified")

                # Draw bounding box and display classification result
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
                draw.text((xmin, ymin), f"{label}: {classification_label}", fill="black", font=font)
                print("box drawn")
            else:
                label = f"{labels[int(class_id)]} ({confidence:.2f})"
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
                draw.text((xmin, ymin), label, fill="red", font=font)

        # Convert the modified image to base64 for display
        img_stream = BytesIO()
        image.save(img_stream, format='PNG')
        img_stream.seek(0)
        img_b64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

        image.save('output_image.png')
        simple_image = Image.new("RGB", (100, 100), "white")
        simple_image.save('simple_image.png')

        return img_b64

    except Exception as e:
        return f"Error processing the image: {str(e)}"

# Homepage and uploading image
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Detect objects and classify "bowl" objects
        img_b64 = detect_and_classify(file)

        return render_template('detect-classify-index.html', image_data=img_b64)

    return render_template('detect-classify-index.html')

if __name__ == '__main__':
    app.run(debug=True)
