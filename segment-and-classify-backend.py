import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from io import BytesIO
import base64

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
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
model.eval()

# Load DeepLabV3 for segmentation
seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
seg_model = seg_model.to(device)
seg_model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to segment and classify food
def segment_and_classify(image_path):
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)

    # Semantic segmentation (DeepLabV3)
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_image)['out'][0]

    output_predictions = torch.argmax(output, dim=0).cpu().numpy()

    # Convert segmentation to bounding boxes and labels
    labels = np.unique(output_predictions)
    segments = {}

    for label in labels:
        mask = (output_predictions == label)
        segment_pixels = np.where(mask)
        segments[label] = segment_pixels

    # Classify each segment and estimate its size
    results = []
    for label, pixels in segments.items():
        y, x = pixels[0], pixels[1]
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)

        cropped_img = original_image[min_y:max_y+1, min_x:max_x+1]
        cropped_img_pil = Image.fromarray(cropped_img)

        cropped_img_tensor = transform(cropped_img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(cropped_img_tensor)
            _, predicted_class = torch.max(output, 1)

        class_label = "Food Class " + str(predicted_class.item())
        segment_area = len(y)  # Pixel count for the segment
        results.append({
            'food_type': class_label,
            'area_pixels': segment_area,
            'segment_coords': (min_x, max_x, min_y, max_y)
        })

    return results, output_predictions, original_image

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

            # Run segmentation and classification
            results, segmentation_mask, original_image = segment_and_classify(file_path)

            # Create output image for visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(original_image)

            for result in results:
                food_type = result['food_type']
                min_x, max_x, min_y, max_y = result['segment_coords']
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(min_x, min_y, f"{food_type}: {result['area_pixels']}px", color="red", fontsize=12)

            # Save the result image to a BytesIO object
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            img_b64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            return render_template('index.html', image_data=img_b64, results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)