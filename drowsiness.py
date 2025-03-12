import os
import gdown
import dlib
import numpy as np
import tensorflow as tf
import cv2
import gradio as gr
import pygame
from scipy.spatial import distance as dist
from collections import deque

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Google Drive File ID for shape_predictor_68_face_landmarks.dat
FILE_ID = "1J3CXKu_o3Bu-U3L1Iy9kESGQhkkF9gW0"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# Download the .dat file if not already present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Initialize pygame mixer for sound alerts
pygame.mixer.init()
ALARM_SOUND_PATH = "alarm.mp3"  # Ensure you upload this file separately

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

# Load dlib's face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(MODEL_PATH)

# Load TensorFlow model
model = tf.keras.models.load_model("drowsiness_model.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Constants
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False
CALIBRATION_FRAMES = 50
calibrated = False
calibrated_threshold = 0.2
ear_values = deque(maxlen=CALIBRATION_FRAMES)

def detect_drowsiness(frame):
    global COUNTER, ALARM_ON, calibrated, calibrated_threshold
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    alert_message = ""
    
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        ear_values.append(ear)

        # Calibration phase
        if not calibrated and len(ear_values) >= CALIBRATION_FRAMES:
            calibrated_threshold = np.mean(ear_values) * 0.8  # 80% of mean EAR
            calibrated = True
            alert_message = f"Calibration completed. EAR threshold set to {calibrated_threshold:.2f}"

        # Prepare image for TensorFlow model
        eye_image = cv2.resize(gray, (64, 64)) / 255.0  # Normalize
        eye_image = np.expand_dims(eye_image, axis=(0, -1))
        prediction = model.predict(eye_image)[0][0]

        if calibrated and (ear < calibrated_threshold and prediction > 0.5):
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                ALARM_ON = True
                alert_message = "🚨 Drowsiness Alert! Wake up!"
                play_alarm()
        else:
            COUNTER = 0
            ALARM_ON = False
            stop_alarm()

        # Draw landmarks
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    return frame, alert_message

def gradio_interface(frame):
    frame, alert = detect_drowsiness(frame)
    return frame, alert

iface = gr.Interface(fn=gradio_interface, 
                     inputs=gr.Image(source="webcam", streaming=True), 
                     outputs=["image", "text"],
                     title="Driver Drowsiness Detection System",
                     description="Detects driver drowsiness using facial landmarks and a deep learning model.")

if __name__ == "__main__":
    iface.launch()
