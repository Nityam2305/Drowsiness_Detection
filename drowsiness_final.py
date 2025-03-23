import os
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
import mediapipe as mp  # Replace dlib with Mediapipe
import pygame
from scipy.spatial import distance as dist
from collections import deque

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize pygame mixer for sound alerts
pygame.mixer.init()
ALARM_SOUND_PATH = "C:\\Users\\bhavi\\OneDrive\\Documents\\Projects\\Drowsiness\\Alarm.mp3"  # Path to your alarm sound file

def play_alarm():
    """Plays alarm sound continuously if not already playing."""
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play(-1)

def stop_alarm():
    """Stops alarm if playing."""
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load TensorFlow model & fix metric warning
model = tf.keras.models.load_model("C:\\Users\\bhavi\\OneDrive\\Documents\\Projects\\Drowsiness\\drowsiness_model.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Constants
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False
CALIBRATION_FRAMES = 50
calibrated = False
calibrated_threshold = 0.2
ear_values = deque(maxlen=CALIBRATION_FRAMES)

# Mediapipe eye landmark indices (adjusted for Face Mesh)
# Left eye: 33 (outer corner), 160 (upper), 158 (upper), 133 (inner corner), 153 (lower), 144 (lower)
# Right eye: 362 (outer corner), 385 (upper), 387 (upper), 263 (inner corner), 373 (lower), 380 (lower)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Streamlit UI
st.title("👁️ Driver Drowsiness Detection System")

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False

# Shutter button to start/stop detection
if st.button("📸 Start/Stop Detection"):
    st.session_state.running = not st.session_state.running

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    frame_display = st.empty()
    st.warning("🔄 Calibrating EAR threshold. Keep your eyes open for a few seconds...")

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video")
            break

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                # Extract left and right eye landmarks
                left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
                right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]

                # Calculate EAR
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)

                # Calibration phase
                if not calibrated and len(ear_values) >= CALIBRATION_FRAMES:
                    calibrated_threshold = np.mean(ear_values) * 0.8  # Set threshold as 80% of mean EAR
                    calibrated = True
                    st.success(f"✅ Calibration completed. EAR threshold set to {calibrated_threshold:.2f}")

                # Prepare image for TensorFlow model
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eye_image = cv2.resize(gray, (64, 64)) / 255.0  # Normalize
                eye_image = np.expand_dims(eye_image, axis=(0, -1))
                prediction = model.predict(eye_image)[0][0]

                # Drowsiness detection logic
                if calibrated and (ear < calibrated_threshold and prediction > 0.5):
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                        ALARM_ON = True
                        st.warning("🚨 Drowsiness Alert! Wake up!")
                        play_alarm()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0  # Properly reset counter if condition is not met
                    ALARM_ON = False
                    stop_alarm()

                # Draw landmarks
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        frame_display.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Ensure the script is run using Streamlit
if __name__ == "__main__":
    st.warning("⚠️ Run this script using: `streamlit run drowsiness.py`")
