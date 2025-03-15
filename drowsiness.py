import os
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Paths to local files
MODEL_PATH = "drowsiness_model.h5"
ALARM_SOUND_PATH = "Alarm.mp3"  # Updated to match the correct file name

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize mediapipe drawing utils (for visualization)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# State management (Streamlit uses session state)
if "COUNTER" not in st.session_state:
    st.session_state["COUNTER"] = 0
if "ALARM_ON" not in st.session_state:
    st.session_state["ALARM_ON"] = False
if "calibrated" not in st.session_state:
    st.session_state["calibrated"] = False
if "calibrated_threshold" not in st.session_state:
    st.session_state["calibrated_threshold"] = 0.2
if "ear_values" not in st.session_state:
    st.session_state["ear_values"] = deque(maxlen=50)

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame):
    if frame is None:
        return None, "No frame received."

    # Convert the frame to RGB for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    alert_message = ""

    # Process the frame with mediapipe
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get image dimensions
            h, w, _ = frame.shape

            # Extract landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])

            # Mediapipe landmark indices for left and right eyes
            left_eye = [
                landmarks[362],  # outer corner
                landmarks[385],  # upper middle
                landmarks[387],  # upper middle
                landmarks[263],  # inner corner
                landmarks[373],  # lower middle
                landmarks[380],  # lower middle
            ]
            right_eye = [
                landmarks[33],   # outer corner
                landmarks[160],  # upper middle
                landmarks[158],  # upper middle
                landmarks[133],  # inner corner
                landmarks[153],  # lower middle
                landmarks[144],  # lower middle
            ]

            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            st.session_state["ear_values"].append(ear)

            # Calibration phase
            if not st.session_state["calibrated"] and len(st.session_state["ear_values"]) >= st.session_state["ear_values"].maxlen:
                st.session_state["calibrated_threshold"] = np.mean(st.session_state["ear_values"]) * 0.8
                st.session_state["calibrated"] = True
                alert_message = f"Calibration completed. EAR threshold set to {st.session_state['calibrated_threshold']:.2f}"

            # Prepare image for TensorFlow model
            eye_image = cv2.resize(gray, (64, 64)) / 255.0
            eye_image = np.expand_dims(eye_image, axis=(0, -1))
            prediction = model.predict(eye_image)[0][0]

            if st.session_state["calibrated"] and (ear < st.session_state["calibrated_threshold"] and prediction > 0.5):
                st.session_state["COUNTER"] += 1
                if st.session_state["COUNTER"] >= 20 and not st.session_state["ALARM_ON"]:
                    st.session_state["ALARM_ON"] = True
                    alert_message = "🚨 Drowsiness Alert! Wake up!"
            else:
                st.session_state["COUNTER"] = 0
                st.session_state["ALARM_ON"] = False

            # Draw landmarks
            for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Draw all face landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    return frame, alert_message

# Streamlit app
st.title("Driver Drowsiness Detection System")
st.write("Detects driver drowsiness using facial landmarks and a deep learning model.")

# Center the video feed and alert
with st.container():
    st.markdown("<style> .centered { display: flex; flex-direction: column; align-items: center; } </style>", unsafe_allow_html=True)
    with st.container():
        FRAME_WINDOW = st.image([], channels="BGR")
        alert_placeholder = st.empty()

    # Use Streamlit's camera input
    camera = st.camera_input("Webcam", label_visibility="hidden")

    while True:
        if camera is not None:
            # Read the image from the camera input
            bytes_data = camera.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                frame, alert = detect_drowsiness(frame)
                FRAME_WINDOW.image(frame, channels="BGR")
                alert_placeholder.write(alert)

                # Play audio in a loop if drowsiness is detected
                if st.session_state["ALARM_ON"]:
                    st.audio(ALARM_SOUND_PATH, format="audio/mp3", start_time=0, loop=True)
                else:
                    st.stop()  # Stop audio when no drowsiness
        else:
            st.write("Please enable the webcam to start detection.")
            break  # Exit loop if no camera input
