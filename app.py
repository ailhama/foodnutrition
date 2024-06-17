from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
import json

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load model CNN
model_path = "./my_model"

# Assuming the model_path is saved in SavedModel format
cnn_model = tf.saved_model.load(model_path)

# Get the inference function
infer = cnn_model.signatures['serving_default']

# Fungsi untuk memuat model Gemini AI dan mendapatkan respons
def get_response_nutrition(image, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([image[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error during API call: {e}")
        return None

# Preprocess data gambar
def prep_image(uploaded_file):
    if uploaded_file is not None:
        # Read the file as bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File is uploaded!")

# Preprocess gambar untuk model CNN
def prep_image_cnn(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Konfigurasi Streamlit App
st.header("Food classification and nutrition prediction with CNN and Generative AI")

# Upload file gambar
upload_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])
if upload_file is not None:
    # Menampilkan gambar yang diupload
    max_width = 200
    max_height = 200
    image = Image.open(upload_file)
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    st.image(image, caption="Gambar berhasil diunggah")

    # Menyiapkan gambar untuk model CNN
    image_for_cnn = prep_image_cnn(image)

    # Mengklasifikasikan gambar menggunakan model CNN
    image_tensor = tf.convert_to_tensor(image_for_cnn, dtype=tf.float32)
    predictions = infer(tf.constant(image_tensor))

    # Using the correct key to access the prediction output
    prediction_key = 'dense_7'
    predictions = predictions[prediction_key].numpy()

    class_idx = np.argmax(predictions, axis=1)[0]
    class_labels = ['Ayam Goreng', 'Bakso', 'Bubur Ayam', 'Mi Goreng', 'Nasi Putih', 'Sate', 'Soto',
                'Telur Dadar', 'Telur Mata Sapi', 'bakwan', 'batagor', 'bihun goreng', 'ca sayur', 
                'cake', 'cumi asam manis', 'cumi goreng tepung', 'dimsum', 'donat', 'gado gado', 
                'ikan goreng', 'kentang goreng', 'martabak', 'mie ayam', 'nasi goreng', 'nasi kuning', 
                'nasi padang', 'pecel', 'pempek', 'pepes ikan', 'perkedel', 'rawon', 'rendang', 
                'salad buah', 'sayur asem', 'singkong goreng', 'sop daging sapi', 'tempe goreng', 
                'tongseng kambing', 'yoghurt']
    class_label = class_labels[class_idx]
    st.write(f"Klasifikasi Makanan: {class_label}")

    # Menyiapkan gambar untuk integrasi dengan model Gemini AI
    image_data = prep_image(upload_file)

    # Prompt Template
    input_prompt_nutrition = f"""
    Anda adalah seorang Ahli Gizi yang ahli. Sebagai ahli gizi yang terampil, Anda diharuskan untuk menganalisis makanan dalam gambar dan menentukan nilai gizi total.
    Gambar ini memperlihatkan {class_label}.
    Silakan berikan rincian dari jenis makanan yang ada dalam {class_label} beserta kandungan gizinya.
    Berikut kata yang harus ditampilkan :
    Ukuran porsi, Kalori, Protein, Lemak, Karbohidrat, Serat
    tampilkan dalam bentuk raw string JSON
    """
    
    # Memuat respons nutrisi secara otomatis
    with st.spinner('Menghitung Nilai Nutrisi...'):
        response = get_response_nutrition(image_data, input_prompt_nutrition)

    if response:
        st.subheader("Nutrisi AI:")
        st.write(response)
    else:
        st.error("Gagal mengambil informasi nutrisi.")