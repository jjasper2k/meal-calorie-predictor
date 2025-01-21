from flask import Flask, render_template, request, redirect
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use YOLOv5 small model

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to perform object detection and draw bounding boxes
def detect_and_draw_boxes(file):
    try:
        # Load the image
        image = Image.open(file).convert("RGB")
        img_array = np.array(image)  # Convert to numpy array for YOLO

        # Perform object detection
        results = yolo_model(img_array)

        # Extract detection results
        detected_items = results.xyxy[0].numpy()  # Bounding box coordinates and confidence
        labels = results.names  # Class labels

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for item in detected_items:
            xmin, ymin, xmax, ymax, confidence, class_id = item
            label = f"{labels[int(class_id)]} ({confidence:.2f})"
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
            draw.text((xmin, ymin), label, fill="red", font=font)

        # Convert the modified image to base64
        img_stream = BytesIO()
        image.save(img_stream, format='PNG')
        img_stream.seek(0)
        img_b64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
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

        # Detect objects and draw bounding boxes
        img_b64 = detect_and_draw_boxes(file)

        return render_template('detect-classify-index.html', image_data=img_b64)

    return render_template('detect-classify-index.html')

if __name__ == '__main__':
    app.run(debug=True)
