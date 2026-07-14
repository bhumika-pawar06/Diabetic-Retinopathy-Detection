import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# 🔥 LIME
from backend_lime import generate_lime_explanation

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DR Detection", layout="centered")

# =========================
# CUSTOM CSS (UI DESIGN)
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #0b3c5d;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4a6fa5;
    margin-bottom: 25px;
}

/* Upload box center */
div[data-testid="stFileUploader"] {
    display: flex;
    justify-content: center;
}

/* Upload styling */
[data-testid="stFileUploader"] {
    border: 2px dashed #0b3c5d;
    border-radius: 10px;
    padding: 10px;
}

/* Center content */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = load_model("hybrid_model_final.h5")

class_names = ["No DR", "Mild", "Moderate", "Severe"]

# =========================
# HEADER
# =========================
st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)

st.markdown("<div class='title'>Diabetic Retinopathy Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Based Retinal Image Analysis</div>", unsafe_allow_html=True)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict & Explain"):

        with st.spinner("Analyzing image..."):

            # =========================
            # Prediction
            # =========================
            img_resized = img.resize((224, 224))
            img_array = keras_image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # =========================
            # LIME
            # =========================
            pred_class_lime, probs, lime_img = generate_lime_explanation(img)

        # =========================
        # OUTPUT
        # =========================
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # =========================
        # LIME OUTPUT
        # =========================
        st.subheader("🟢 LIME Explanation")
        st.image((lime_img * 255).astype("uint8"), use_container_width=True)

        # =========================
        # EXTRA INFO
        # =========================
        with st.expander("🔍 Severity Explanation"):
            st.write(f"{predicted_class} stage detected based on retinal features.")

        with st.expander("⚕ Medical Advice"):
            st.write("Consult an ophthalmologist and maintain proper blood sugar levels.")

        st.success("Analysis Completed Successfully!")