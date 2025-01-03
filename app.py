from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = load_model('pneumonia_detection_model5.h5')
    print("Model loaded successfully")
except ValueError as e:
    print("Error loading model:", e)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Define a function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request for prediction")
    if 'file' not in request.files:
        print("No file provided")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    img_path = f'uploads/{file.filename}'
    file.save(img_path)

    img = preprocess_image(img_path)
    prediction = model.predict(img)

    result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
    confidence = float(prediction[0][0])

    print(f"Prediction: {result}, Confidence: {confidence}")
    return jsonify({'prediction': result, 'confidence': confidence})

# Define the main route to render the HTML
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8000,host="0.0.0.0")
