from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- LOAD MODEL .H5 ---
# Pastikan nama file sesuai dengan yang ada di folder Anda
model = tf.keras.models.load_model("model_xception.h5")

# --- LOAD OOD MODEL (MobileNetV2) ---
# Digunakan sebagai filter untuk mendeteksi apakah gambar adalah kucing/anjing atau bukan
ood_model = tf.keras.applications.MobileNetV2(weights='imagenet')

def is_cat_or_dog(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = ood_model.predict(img_array)
    top_class = np.argmax(preds[0])
    
    # Di ImageNet: Anjing (151-268), Kucing (281-285)
    if (151 <= top_class <= 268) or (281 <= top_class <= 285):
        return True
    return False
# Kategori prediksi sesuai dengan urutan abjad folder (cats=0, dogs=1)
CLASS_NAMES = ["Kucing", "Anjing"]

def preprocess_image(image):
    # Xception membutuhkan ukuran gambar 299x299
    image = image.resize((299, 299))
    image = np.array(image) / 255.0 # Normalisasi
    image = np.expand_dims(image, axis=0) # Menambah dimensi batch
    return image

@app.route("/")
def home():
    # Mengarahkan ke halaman UI yang sudah diperbagus sebelumnya
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Cek dulu menggunakan filter MobileNetV2
    if not is_cat_or_dog(image):
        return jsonify({"prediction": "Tidak dapat diprediksi", "confidence": 0.0})
        
    # Proses gambar dan lakukan prediksi dengan model .h5
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    
    confidence = float(np.max(prediction))
    result = CLASS_NAMES[predicted_class_index]
    
    return jsonify({"prediction": result, "confidence": confidence})

if __name__ == "__main__":
    # Gunakan host 0.0.0.0 agar bisa diakses saat di-deploy
    app.run(host="0.0.0.0", port=5000, debug=False)