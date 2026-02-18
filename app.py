import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "coriander_parsley_model.keras",
        compile=False
    )

model = load_my_model()

st.title("🌿 Coriander vs Parsley Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((250, 250))  # ⚠️ taille selon ton modèle
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_names = ["Coriander", "Parsley"]
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
