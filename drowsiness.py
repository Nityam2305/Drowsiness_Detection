import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import pygame
import gdown
import subprocess
from scipy.spatial import distance as dist
from collections import deque

# Attempt to install dlib if missing
try:
    import dlib
except ImportError:
    st.warning("⚠️ dlib is missing! Attempting to install...")
    subprocess.run(["pip", "install", "dlib"], check=True)
    import dlib  # Retry import

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize pygame for sound alerts
pygame.mixer.init()
ALARM_SOUND_PATH = "assets/alarm.mp3"  # Relative path for deployment

# Google Drive File ID for .dat file
FILE_ID = "1J3CXKu_o3Bu-U3L1Iy9kESGQhkkF9gW0"
DEST_PATH = "models/shape_predictor_68_face_landmarks.dat"

# Download the .dat file if not exists
if not os.path.exists(DEST_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", DEST_PATH, quiet=False)

# Function to play/stop alarm
def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load face detector & landmark predictor
try:
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(DEST_PATH)
    use_dlib = True
except Exception as e:
    st.warning(f"⚠️ dlib failed to initialize: {e}. Using OpenCV instead.")
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    use_dlib = False

# Load TensorFlow model
model = tf.keras.models.load_model("models/drowsiness_model.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Constants
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False
CALIBRATION_FRAMES = 50
calibrated = False
calibrated_threshold = 0.2
ear_values = deque(maxlen=CALIBRATION_FRAMES)

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
    st.warning("🔄 Calibrating EAR threshold. Keep your eyes open for a few seconds...")

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if use_dlib:
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

        else:
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eye_image = cv2.resize(roi_gray, (64, 64)) / 255.0
                eye_image = np.expand_dims(eye_image, axis=(0, -1))
                prediction = model.predict(eye_image)[0][0]

                if prediction > 0.5:  
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                        ALARM_ON = True
                        st.warning("🚨 Drowsiness Alert! Wake up!")
                        play_alarm()
                else:
                    COUNTER = 0
                    ALARM_ON = False
                    stop_alarm()

        frame_display.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

st.markdown("---")
st.markdown("### ℹ️ How It Works")
st.markdown("- **EAR (Eye Aspect Ratio)** (if `dlib` works) measures blinking patterns.")
st.markdown("- **Deep Learning Model:** Detects drowsiness based on eye images.")
st.markdown("- **Alarm System:** Plays an alert if drowsiness is detected.")
