from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Ear_Disease_model.h5")  # Ensure the model file is present

# Labels
class_labels = ['Normal', 'Infected']

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Normalize for ResNet50
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)  # Save the uploaded image
            
            # Preprocess image and make prediction
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)[0]
            predicted_class = class_labels[int(prediction > 0.5)]  # Assuming binary classification

            return render_template("index.html", prediction=predicted_class, image_url=file_path)
    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
