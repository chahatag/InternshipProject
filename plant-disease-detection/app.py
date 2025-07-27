import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load model and class labels
model = tf.keras.models.load_model("C:/Users/chaha/plant_disease_model.h5")
with open("C:/Users/chaha/class_indices.json", "r") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# Streamlit app
st.set_page_config(page_title="Plant Disease Detector")
st.title("Plant Disease Detection from Leaf Image")
st.write("Upload a leaf image to identify the disease")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f" Predicted: **{predicted_class.replace('___', ' > ').replace('_', ' ')}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
