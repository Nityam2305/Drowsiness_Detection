import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
from scipy.spatial import distance as dist
from collections import deque

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ALARM_SOUND_PATH = "assets/alarm.mp3"  # Relative path for deployment

# Function to play/stop alarm using JavaScript

def play_alarm():
    """Plays alarm in a loop using JavaScript in Streamlit."""
    alarm_html = f"""
        <audio id="alarmSound" autoplay loop>
            <source src="{ALARM_SOUND_PATH}" type="audio/mp3">
        </audio>
        <script>
            var audio = document.getElementById("alarmSound");
            audio.play();
        </script>
    """
    st.markdown(alarm_html, unsafe_allow_html=True)

def stop_alarm():
    """Stops the alarm by removing the audio element via JavaScript."""
    stop_html = """
        <script>
            var audio = document.getElementById("alarmSound");
            if (audio) { audio.pause(); audio.currentTime = 0; }
        </script>
    """
    st.markdown(stop_html, unsafe_allow_html=True)

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load OpenCV’s face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load TensorFlow model
MODEL_PATH = "models/drowsiness_model.h5"
MODEL_FILE_ID = "1bYj8W6N5u1wnRZ64kZtg7r8V393naGfz"  # Google Drive File ID

# Auto-Download Model if Missing
# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.info("Downloading Drowsiness Model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Constants
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

# Streamlit UI
st.set_page_config(page_title="Drowsiness Detector", page_icon="🚗", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("#### Stay alert while driving. This tool helps detect drowsiness in real-time.")

# Camera Selection
camera_option = st.selectbox("📷 Select Camera", ["Back Camera", "Front Camera"])
camera_index = 0 if camera_option == "Back Camera" else 1

# Start/Stop Button
if "running" not in st.session_state:
    st.session_state.running = False
if st.button("📸 Start/Stop Detection"):
    st.session_state.running = not st.session_state.running

# Video Processing
if st.session_state.running:
    cap = cv2.VideoCapture(camera_index)
    frame_display = st.empty()
    st.warning("🔄 Processing... Please wait.")

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            # Prepare image for TensorFlow model
            eye_image = cv2.resize(roi_gray, (64, 64)) / 255.0
            eye_image = np.expand_dims(eye_image, axis=(0, -1))
            prediction = model.predict(eye_image)[0][0]

            if prediction > 0.5:  # Drowsiness detected
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                    ALARM_ON = True
                    st.warning("🚨 Drowsiness Alert! Wake up!")
                    play_alarm()
            else:
                COUNTER = 0
                if ALARM_ON:
                    ALARM_ON = False
                    stop_alarm()

        frame_display.image(frame, channels="BGR")

    cap.release()
    


st.markdown("---")
st.markdown("### ℹ️ How It Works")
st.markdown("- **Deep Learning Model:** Detects drowsiness based on eye images.")
st.markdown("- **Alarm System:** Plays an alert if drowsiness is detected.")
