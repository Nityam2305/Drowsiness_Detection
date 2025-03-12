import os
import dlib
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
import pygame
import requests
from scipy.spatial import distance as dist
from collections import deque

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize pygame for sound alerts
pygame.mixer.init()

# File URLs (Google Drive direct download links)
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_H5_FILE_ID"
LANDMARK_URL = "https://drive.google.com/uc?export=download&id=YOUR_DAT_FILE_ID"

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "drowsiness_model.h5")
LANDMARK_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "Alarm.mp3")

# Function to download files if not present
def download_file(url, path):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {os.path.basename(path)}..."):
            response = requests.get(url, stream=True)
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            st.success(f"{os.path.basename(path)} downloaded!")

# Download model and landmark files
download_file(MODEL_URL, MODEL_PATH)
download_file(LANDMARK_URL, LANDMARK_PATH)

# Load dlib's face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(LANDMARK_PATH)

# Load TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# EAR Calculation Function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Streamlit UI
st.set_page_config(page_title="🚘 Drowsiness Detection", layout="wide", page_icon="😴")

# Title & Description
st.title("🚘 Driver Drowsiness Detection System")
st.markdown("**🔍 Real-time monitoring to prevent accidents due to drowsiness.**")

# Sidebar Instructions
st.sidebar.header("⚙️ How to Use")
st.sidebar.info(
    "1️⃣ Click 'Start Detection' to begin. \n"
    "2️⃣ Keep your eyes open for calibration. \n"
    "3️⃣ If drowsy, an alarm will sound. \n"
    "4️⃣ Click 'Stop Detection' to exit."
)

# Start/Stop Button
if "running" not in st.session_state:
    st.session_state.running = False

if st.button("🚦 Start/Stop Detection"):
    st.session_state.running = not st.session_state.running

# Constants
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False
CALIBRATION_FRAMES = 50
calibrated = False
calibrated_threshold = 0.2
ear_values = deque(maxlen=CALIBRATION_FRAMES)

# Function to play/stop alarm
def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Video Processing
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    frame_display = st.empty()
    st.warning("🔄 Calibrating EAR threshold. Keep your eyes open...")

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            ear_values.append(ear)

            # Calibration Phase
            if not calibrated and len(ear_values) >= CALIBRATION_FRAMES:
                calibrated_threshold = np.mean(ear_values) * 0.8
                calibrated = True
                st.success(f"✅ Calibration complete. EAR threshold: {calibrated_threshold:.2f}")

            # Prepare image for TensorFlow model
            eye_image = cv2.resize(gray, (64, 64)) / 255.0
            eye_image = np.expand_dims(eye_image, axis=(0, -1))
            prediction = model.predict(eye_image)[0][0]

            # Drowsiness Detection
            if calibrated and (ear < calibrated_threshold and prediction > 0.5):
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                    ALARM_ON = True
                    st.warning("🚨 Drowsiness Alert! Wake up!")
                    play_alarm()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
                stop_alarm()

            # Draw landmarks
            for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Display the frame
        frame_display.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 Developed by Nityam | 🔗 [GitHub](https://github.com/Nityam2305/Drowsiness_Detection)")
