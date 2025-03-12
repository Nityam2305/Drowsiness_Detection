import streamlit as st
# Streamlit UI
st.set_page_config(page_title="Drowsiness Detector", page_icon="🚗", layout="wide")import os
import cv2
import numpy as np
import tensorflow as tf
import gdown
import threading
import time
from scipy.spatial import distance as dist
from collections import deque

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ALARM_SOUND_PATH = "assets/alarm.mp3"  # Relative path for deployment
MODEL_PATH = "models/drowsiness_model.h5"
MODEL_FILE_ID = "1bYj8W6N5u1wnRZ64kZtg7r8V393naGfz"  # Google Drive File ID

# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Auto-Download Model if Missing
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading Drowsiness Model... (May take a few minutes)")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Function to play/stop alarm using JavaScript
def play_alarm():
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

# Constants
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

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

# Start Camera in a Separate Thread
def initialize_camera():
    global cap
    cap = cv2.VideoCapture(camera_index)
    time.sleep(2)  # Allow camera to warm up

camera_thread = threading.Thread(target=initialize_camera)
camera_thread.start()

# Video Processing
if st.session_state.running:
    frame_display = st.empty()
    st.warning("🔄 Processing... Please wait.")

    skip_frames = 2  # Process every 2nd frame to reduce lag
    frame_count = 0

    while cap.isOpened() and st.session_state.running:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video")
            break

        if frame_count % skip_frames != 0:
            continue  # Skip frames to improve speed

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
