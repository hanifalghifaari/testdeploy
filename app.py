import os
import time
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
import google.generativeai as genai
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.cloud.secretmanager as secretmanager
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

app = Flask(__name__)
# Load models
MODEL_PATH = "model/model_xception.keras"
MODEL_PATH = tf.keras.models.load_model(MODEL_PATH, compile=False)


app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Class labels
class_names = ['Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng',
               'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon',
               'Rendang', 'Sate', 'Soto Ayam']

# Utility function to check file extension


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Preprocess image function


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(300, 300))
    img_array = img_to_array(img)  # Konversi gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    img_array = xception_preprocess_input(
        img_array)
    return img_array

# Decode prediction function


def decode_prediction(prediction):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]  # Ambil nama kelas
    confidence = prediction[0][predicted_class_index] * \
        100
    confidence = round(confidence, 1)
    return predicted_class, confidence


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No image provided",
            },
            "data": None
        }), 400

    image = request.files["image"]
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(file_path)  # Simpan gambar

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Start timing prediction
        start_predict_time = time.time()

        # Predict the image
        prediction = MODEL_PATH.predict(img_array)
        predicted_class, confidence = decode_prediction(prediction)
        # End timing prediction
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        # Remove the saved image after prediction
        os.remove(file_path)

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success",
            },
            "data": {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "predict_time": predict_time
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Invalid file type",
            },
            "data": None
        }), 400


if __name__ == "__main__":
    # Create upload folder if not exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
