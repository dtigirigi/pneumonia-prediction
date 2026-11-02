import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)

# Load CNN model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/cnn_pneumonia_model.keras")
    return model

model = load_model()

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict the probability of pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width="content")

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prob = float(model.predict(img)[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

    # Display results
    st.markdown(f"### **Prediction:** {label}")
    st.markdown(f"**Probability of Pneumonia:** {prob:.4f}")
    st.markdown(f"**Confidence Level:** {confidence*100:.2f}%")

    # Confidence interpretation
    if prob >= 0.85:
        st.success("High confidence in Pneumonia prediction.")
    elif prob >= 0.5:
        st.warning("Moderate probability of Pneumonia. Clinical correlation advised.")
    else:
        st.info("Likely Normal. Continue clinical evaluation if symptoms persist.")

    st.write("This prediction is based on the finalized CNN model trained on chest X-ray images.")