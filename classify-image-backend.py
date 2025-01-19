import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained ResNet-18 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_classes = 101  
model.fc = nn.Linear(model.fc.in_features, num_classes)

script_dir = os.path.dirname(__file__)
checkpoint_path = os.path.join(script_dir, "checkpoints", "best_model.pth")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to classify the entire image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    class_label = "Food Class " + str(predicted_class.item())
    return class_label

# Homepage and uploading image
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Run classification
            class_label = classify_image(file_path)

            # Prepare the image for display
            image = Image.open(file_path)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.array(image))
            ax.axis('off')
            ax.set_title(f"Prediction: {class_label}", fontsize=16, color="blue")

            # Save the result image to a BytesIO object
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            img_b64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            return render_template('index.html', image_data=img_b64, prediction=class_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)