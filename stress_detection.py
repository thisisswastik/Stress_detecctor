import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained VGG-16 model
MODEL_PATH = "stress_detector.h5"
model = load_model(MODEL_PATH)

# Class labels
class_labels = ["Low Stress", "Medium Stress", "High Stress"]

# Function to preprocess an image for VGG-16
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to VGG-16 input size
    image = image.convert("RGB")  # Convert grayscale to RGB
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict_stress(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    stress_level = np.argmax(prediction)
    return class_labels[stress_level], prediction[0][stress_level] * 100

# Function to process real-time webcam frames
def detect_stress_in_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to VGG-16 input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = img_to_array(frame) / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    prediction = model.predict(frame)
    stress_level = np.argmax(prediction)
    return class_labels[stress_level], prediction[0][stress_level] * 100

# UI Layout
st.title("🧠 Stress Level Detection")
st.write("Upload an image or use the real-time camera to detect stress levels.")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Stress Level
    stress_label, confidence = predict_stress(image)
    st.subheader(f"Predicted Stress Level: {stress_label} ({confidence:.2f}%)")

    # Stress Recommendations
    st.write("### Recommended Exercises:")
    if stress_label == "Low Stress":
        st.success("✅ Keep up your healthy lifestyle! Try mindfulness meditation or light yoga.")
    elif stress_label == "Medium Stress":
        st.warning("⚠️ Consider deep breathing exercises, listening to relaxing music, or a short walk.")
    else:
        st.error("🚨 High stress detected! Try progressive muscle relaxation, guided meditation, or talking to a friend.")

# Webcam Support - Capture Image
st.write("### 📷 Capture Image from Webcam")
if st.button("Open Webcam"):
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Predict Stress Level
        stress_label, confidence = predict_stress(image)
        st.subheader(f"Predicted Stress Level: {stress_label} ({confidence:.2f}%)")

        # Stress Recommendations
        st.write("### Recommended Exercises:")
        if stress_label == "Low Stress":
            st.success("✅ Keep up your healthy lifestyle! Try mindfulness meditation or light yoga.")
        elif stress_label == "Medium Stress":
            st.warning("⚠️ Consider deep breathing exercises, listening to relaxing music, or a short walk.")
        else:
            st.error("🚨 High stress detected! Try progressive muscle relaxation, guided meditation, or talking to a friend.")

# Real-Time Webcam Detection
st.write("### 🎥 Real-Time Stress Detection")
run_webcam = st.checkbox("Enable Real-Time Detection")

if run_webcam:
    st.warning("Press 'q' to exit the real-time detection window.")
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        stress_label, confidence = detect_stress_in_frame(frame)

        # Display the result on the frame
        cv2.putText(frame, f"Stress: {stress_label} ({confidence:.2f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Real-Time Stress Detection", frame)

        # Press 'q' to exit the real-time window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
