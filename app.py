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

app = Flask(__name__)

# Mengakses API Key dari Secret Manager


def get_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    project_id = 'your-project-id'
    secret_version = f'projects/{project_id}/secrets/{secret_name}/versions/latest'
    response = client.access_secret_version(name=secret_version)
    return response.payload.data.decode('UTF-8')


# Gantilah 'my-api-key' dengan nama secret Anda
API_KEY = get_secret('my-api-key')

# Load models
MODEL_PATH_XCEPTION = "model/model_xception.keras"
MODEL_PATH_INCEPTION = "model\InceptionV3.keras"
model_xception = tf.keras.models.load_model(MODEL_PATH_XCEPTION, compile=False)
model_inception = tf.keras.models.load_model(
    MODEL_PATH_INCEPTION, compile=False)

# Configure the API key
genai.configure(api_key=API_KEY)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,)
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "sate",
            ],
        },
        {
            "role": "model",
            "parts": [
                "**Makanan:** Sate\n\n**Bahan-bahan:**\n- Daging sapi (non-veg)\n- Bawang merah (veg)\n- Bawang putih (veg)\n- Ketumbar (veg)\n- Jinten (veg)\n- Merica (veg)\n- Kunyit (veg)\n- Jahe (veg)\n- Kecap manis (veg)\n\n**Rincian Emisi Karbon:**\n\n| Bahan | Pertanian | Pengolahan | Transportasi | Ritel | Pengemasan | Kerugian | Penggunaan Lahan |\n|---|---|---|---|---|---|---|---|\n| Daging sapi | 21.92 | 1.55 | 0.59 | 0.25 | 0.37 | 3.85 | 1.27 |\n| Bawang merah | 0.21 | 0.0 | 0.09 | 0.04 | 0.04 | 0.1 | 0.0 |\n| Bawang putih | 0.28 | 0.0 | 0.09 | 0.02 | 0.05 | 0.08 | 0.0 |\n| Ketumbar | 1.58 | 0.41 | 0.13 | 0.05 | 0.11 | 0.47 | 0.49 |\n| Jinten | 3.37 | 0.05 | 0.11 | 0.04 | 0.12 | -0.01 | -3.26 |\n| Merica | 6.69 | 0.33 | 0.11 | 0.04 | 0.72 | 12.94 | 25.81 |\n| Kunyit | 0.49 | 0.79 | 0.18 | 0.27 | 0.18 | 0.29 | 0.96 |\n| Jahe | 0.71 | 0.01 | 0.18 | 0.02 | 0.15 | 0.66 | 0.37 |\n| Kecap manis | - | - | - | - | - | - | - |\n\n**Total Emisi Karbon per Porsi:** 35,42 kg CO2e\n\n**Saran Keberlanjutan:**\n\n* 🌱 Kurangi konsumsi daging sapi, karena merupakan kontributor utama emisi karbon.\n* 🌍 Beli bumbu dan rempah-rempah dari sumber lokal untuk mengurangi emisi transportasi.\n* 🍃 Pilih kecap manis organik yang diproduksi dengan práticas berkelanjutan.\n* ♻️ Gunakan tusuk sate yang dapat digunakan kembali untuk mengurangi kemasan.\n\n**Fakta Keberlanjutan:**\n\n* Produksi daging sapi bertanggung jawab atas sekitar 18% emisi gas rumah kaca global.\n* Mengurangi konsumsi daging adalah salah satu cara paling efektif untuk mengurangi jejak karbon makanan individu.\n\n**Penutupan dengan Kutipan:**\n\n\"Makanan yang kita makan memiliki dampak yang signifikan terhadap lingkungan kita. Dengan membuat pilihan makanan yang berkelanjutan, kita dapat memainkan peran penting dalam mengurangi jejak karbon kita dan melindungi planet ini.\"",
            ],
        },
        {
            "role": "user",
            "parts": [
                "Pizza",
            ],
        },
        {
            "role": "model",
            "parts": [
                "**Makanan:** Pizza\n\n**Bahan-bahan:**\n- Tepung terigu (veg)\n- Saus tomat (veg)\n- Keju (non-veg)\n- Sosis (non-veg)\n- Jamur (veg)\n- Bawang bombay (veg)\n- Paprika (veg)\n- Zaitun (veg)\n\n**Rincian Emisi Karbon:**\n\n| Bahan | Pertanian | Pengolahan | Transportasi | Ritel | Pengemasan | Kerugian | Penggunaan Lahan |\n|---|---|---|---|---|---|---|---|\n| Tepung terigu | 0.82 | 0.21 | 0.13 | 0.06 | 0.09 | 0.18 | 0.1 |\n| Saus tomat | 0.71 | 0.01 | 0.18 | 0.02 | 0.15 | 0.66 | 0.37 |\n| Keju | 13.1 | 0.74 | 0.14 | 0.33 | 0.17 | 2.58 | 4.47 |\n| Sosis | 21.92 | 1.55 | 0.59 | 0.25 | 0.37 | 3.85 | 1.27 |\n| Jamur | 0.49 | 0.79 | 0.18 | 0.27 | 0.18 | 0.29 | 0.96 |\n| Bawang bombay | 0.21 | 0.0 | 0.09 | 0.04 | 0.04 | 0.1 | 0.0 |\n| Paprika | 0.72 | 0.0 | 0.1 | 0.04 | 0.04 | 0.08 | 0.0 |\n| Zaitun | 3.37 | 0.05 | 0.11 | 0.04 | 0.12 | -0.01 | -3.26 |\n\n**Total Emisi Karbon per Porsi (4 potong):** 44,8 kg CO2e\n\n**Saran Keberlanjutan:**\n\n* 🌱 Pilih pizza vegetarian dengan topping sayuran segar seperti jamur, paprika, dan bawang bombay.\n* 🌍 Beli bahan-bahan dari sumber lokal untuk mengurangi emisi transportasi.\n* 🍃 Gunakan tepung organik dan saus tomat yang diproduksi secara berkelanjutan.\n* ♻️ Gunakan kemasan ramah lingkungan untuk sisa pizza.\n\n**Fakta Keberlanjutan:**\n\n* Produksi daging sapi merupakan kontributor utama emisi gas rumah kaca, sehingga mengurangi konsumsi daging dapat secara signifikan mengurangi jejak karbon makanan.\n* Memilih bahan makanan lokal dapat membantu mengurangi emisi transportasi dan mendukung ekonomi lokal.\n\n**Penutupan dengan Kutipan:**\n\n\"Setiap gigitan yang kita ambil mempunyai dampak pada planet kita. Dengan membuat pilihan makanan yang berkelanjutan, kita dapat mengurangi jejak karbon kita dan memastikan masa depan yang lebih berkelanjutan.\"",
            ],
        },
    ]
)

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


def preprocess_image(image_path, model_name):
    img = load_img(image_path, target_size=(299, 299)
                   if model_name == "inceptionV3" else (300, 300))
    img_array = img_to_array(img)  # Konversi gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    if model_name == "inceptionV3":
        img_array /= 255.0  # Preprocessing sesuai InceptionV3
    else:
        img_array = xception_preprocess_input(
            img_array)  # Preprocessing sesuai Xception
    return img_array

# Decode prediction function


def decode_prediction(prediction):
    # Ambil indeks kelas dengan probabilitas tertinggi
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
    model_choice = request.form.get(
        'model', 'inception')
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(file_path)  # Simpan gambar

        # Preprocess the image
        img_array = preprocess_image(file_path, model_choice)

        # Choose model based on user input
        if model_choice == "inceptionV3":
            model = model_inception
        else:
            model = model_xception

        # Start timing prediction
        start_predict_time = time.time()

        # Predict the image
        prediction = model.predict(img_array)
        predicted_class, confidence = decode_prediction(prediction)
        text = f"""**Makanan:** {predicted_class}\n\n
        Anda adalah asisten pintar berbahasa Indonesia yang ahli dalam menghitung jejak karbon dari makanan atau struk makanan dan breakdown bahan bahan pada makanan nya lalu cocokan bahan bahan itu dengan yang ada dalam data emisi yg saya berikan di bawah. Berikut adalah instruksi untuk tugas Anda:\n\n
        1. **Identifikasi Makanan**:
           - Tentukan apakah item makanan bersifat vegetarian atau non-vegetarian.
           - Buat daftar bahan makanan beserta klasifikasinya (veg/non-veg) yang ada dalam data karbon emisi yg saya berikan saja yg kamu tampilkan.\n\n
        2. **Rincian Emisi Karbon**:
           - Hitung emisi karbon berdasarkan faktor seperti pertanian, pengolahan, transportasi, ritel, pengemasan, kerugian, dan penggunaan lahan.
           - Daftar bahan makanan yang ada dalam data karbon emisi yang saya berikan saja yang kamu tampilkan.\n\n
        3. **Rincian Emisi Karbon per Makanan**:
           - Jika ada beberapa jenis makanan, beri rincian emisi karbon secara terpisah.
           - Tampilkan total emisi karbon dalam tabel ringkas.\n\n
        4. **Saran Keberlanjutan**:
           - Berikan saran terkait keberlanjutan dan pengurangan jejak karbon, seperti:
             - 🌱 Belilah makanan secukupnya.
             - 🌍 Pilih produk lokal.
             - 🍃 Pilih makanan organik.
             - ♻️ Gunakan kemasan ramah lingkungan.\n\n
        5. **Fakta Keberlanjutan**:
           - Sertakan fakta penting tentang jejak karbon makanan (contoh: "Tahu memiliki jejak karbon 10x lebih rendah daripada daging sapi.")\n\n
        6. - Berikan kutipan yang menginspirasi tentang pentingnya mengurangi jejak karbon makanan untuk perubahan iklim.\n\n
        **Referensi Data untuk Perhitungan Emisi**:
        (Data emisi karbon seperti apel, pisang, daging sapi, dsb.)
        """
        responsegenai = chat_session.send_message(text)
        generated_text = responsegenai.text
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
                "GenText": generated_text,
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
