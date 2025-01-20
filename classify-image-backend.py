from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained ResNet-18 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_classes = 101  
model.fc = nn.Linear(model.fc.in_features, num_classes)

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

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to classify the uploaded image
def classify_image(image_file):
    image = Image.open(image_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    # Map predicted class index to class label
    predicted_class_label = class_labels[predicted_class.item()]
    return predicted_class_label

# Homepage and uploading image
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Classify the uploaded image directly
            class_label = classify_image(file)

            # Convert image to base64 for display
            image = Image.open(file).convert("RGB")
            img_stream = BytesIO()
            image.save(img_stream, format='PNG')
            img_stream.seek(0)
            img_b64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            return render_template('classify-index.html', image_data=img_b64, prediction=class_label)

    return render_template('classify-index.html')

if __name__ == '__main__':
    app.run(debug=True)