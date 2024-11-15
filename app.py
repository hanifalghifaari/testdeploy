import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io

app = Flask(__name__)

# Load model
MODEL_PATH = "model/Densenet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

app.config["ALLOWED_EXTENSIONS"] = set(["jpg", "jpeg", "png"])
app.config["UPLOAD_FOLDER"] = "static/uploads"

with open("labels.txt", "r") as file:
    class_labels = file.read().splitlines()


def allowed_file(filename):
    return '.' in filename and \
        filename.split('.', 1)[1] in app.config["ALLOWED_EXTENSIONS"]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200


@app.route("/predict_image", methods=["POST", "GET"])
def predict_image():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # prerpocessing the image
            img = load_img(image_path, target_size=(224, 224))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Rescale image

            # predicting the image
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            return jsonify(
                {
                    "status": {
                        "code": 200,
                        "message": "Success",
                    },
                    "data": {
                        "predicted_class": predicted_class_label,
                        "confidence": float(confidence)
                    }
                }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad Request",
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method Not Allowed",
            },
            "data": None
        }), 405


if __name__ == "__main__":
    app.run()
