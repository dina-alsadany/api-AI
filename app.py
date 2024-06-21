from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Paths to the model directories
xray_model_dir = 'model/xray'
ct_model_dir = 'model/ct'

# Check if the model directories exist
if not os.path.exists(xray_model_dir):
    raise FileNotFoundError(f"X-ray model directory not found at {xray_model_dir}")
if not os.path.exists(ct_model_dir):
    raise FileNotFoundError(f"CT model directory not found at {ct_model_dir}")

# Load the pre-trained models
xray_model_path = os.path.join(xray_model_dir, 'model.h5')
ct_model_path = os.path.join(ct_model_dir, 'model.h5')

# Ensure the models' input shapes are correct
xray_model = load_model(xray_model_path)
ct_model = load_model(ct_model_path)

# Function to preprocess the image
def preprocess_image(image, input_shape):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(input_shape)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Categories that the models predict
categories_xray = [
    'Atelectasis', 
    'Cardiomegaly', 
    'Consolidation',
    'Edema', 
    'Effusion',
    'Emphysema', 
    'Fibrosis',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax', 
    'Normal'
]

categories_ct = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

@app.route('/xray/predict', methods=['POST'])
def predict_xray():
    return predict(xray_model, categories_xray, (224, 224))  # Adjust input shape if necessary

@app.route('/ct/predict', methods=['POST'])
def predict_ct():
    return predict(ct_model, categories_ct, (224, 224))  # Adjust input shape if necessary

def predict(model, categories, input_shape):
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image, input_shape)
        predictions = model.predict(processed_image)[0]
        results = {category: float(prediction) * 100 for category, prediction in zip(categories, predictions)}
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'An error occurred during prediction. Please check the server logs for more details.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
