import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model_path = r"C:\Users\Goutham C S\Desktop\model2\plantvillage_multiclass_model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels (must match dataset folder names)
class_labels = list(model.classes_) if hasattr(model, "classes_") else None
if class_labels is None:
    # Fallback: define manually based on your dataset
    class_labels = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot", "Tomato___healthy"
        # add all others here...
    ]

def predict_plant_health(image):
    image = image.resize((128, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return class_labels[class_idx], confidence

# Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to check for diseases.")

uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing...")
    label, confidence = predict_plant_health(image)

    st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")

    if "healthy" in label.lower():
        st.info("✅ Plant is healthy. No action needed.")
    else:
        st.warning("⚠️ Plant shows signs of disease. Consider appropriate treatment.")
