from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model Xception yang sudah dilatih
model = tf.keras.models.load_model("model_xception.h5")

# Kategori prediksi (Sesuaikan dengan dataset Anda)
CLASS_NAMES = ["kucing", "anjing"]

def preprocess_image(image):
    # Xception membutuhkan ukuran gambar 299x299[cite: 1]
    image = image.resize((299, 299))
    image = np.array(image) / 255.0 # Normalisasi
    image = np.expand_dims(image, axis=0) # Menambah dimensi batch
    return image

@app.route("/")
def home():
    # Halaman utama untuk mengunggah gambar[cite: 1]
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Pemrosesan gambar menggunakan model transfer learning XCeption[cite: 1]
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    
    result = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(prediction))
    
    return jsonify({"prediction": result, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)