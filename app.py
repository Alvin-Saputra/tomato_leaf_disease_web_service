from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Muat model
MODEL_PATH = 'model_daun.h5'  # Ubah ke path modelmu
model = load_model(MODEL_PATH)

# Label kelas (ubah sesuai modelmu)
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'] # contoh label


# Fungsi preprocessing gambar
def preprocess_image(img, target_size=(128, 128)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimensi
    return img_array

# Endpoint klasifikasi gambar
@app.route('/classify-image', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(img_file.read()))
        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)
        predicted_index = np.argmax(prediction[0])
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction[0]))

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route dasar untuk tes
@app.route('/')
def home():
    # return render_template('index.html')  # Bisa diganti dengan "return 'Hello, World!'" jika tidak pakai template
    return 'Hello, World!'

# Contoh endpoint API POST
@app.route('/api/echo', methods=['POST'])
def api_echo():
    data = request.json
    return jsonify({'you_sent': data})

if __name__ == '__main__':
    app.run(debug=True)
