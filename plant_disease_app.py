import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model_path = r"C:\Users\Goutham C S\Desktop\model2\plantvillage_multiclass_model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels (38 from PlantVillage dataset)
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

num_classes = model.output_shape[-1]

# Sanity check
if len(class_labels) != num_classes:
    st.error(f"⚠️ Mismatch: Model outputs {num_classes} classes, "
             f"but class_labels has {len(class_labels)} entries.")
    st.stop()

def predict_plant_health(image):
    """Preprocess image and predict class"""
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
