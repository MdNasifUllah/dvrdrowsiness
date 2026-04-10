import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import time

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("8 SIG BN || 24 INF DIV")

# Initialize MediaPipe
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_face_mesh()

# Eye indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    if len(eye_landmarks) != 6:
        return 0.0
    
    vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if horizontal == 0:
        return 0.0
    
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    ear_threshold = st.slider("EAR Threshold", 0.15, 0.35, 0.22, 0.01)
    frame_threshold = st.slider("Frame Threshold", 10, 60, 30, 5)
    st.markdown("---")
    st.info("💡 Keep your face clearly visible in good lighting")

# Camera input
camera_image = st.camera_input("Position your face in frame", key="camera")

if camera_image is not None:
    # Read image
    image = Image.open(camera_image)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Metrics
    left_ear = 0.0
    right_ear = 0.0
    avg_ear = 0.0
    status = "NO FACE DETECTED"
    status_color = "red"
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get eye points
        left_points = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE_INDICES])
        right_points = np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE_INDICES])
        
        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)
        avg_ear = (left_ear + right_ear) / 2
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Determine status
        if avg_ear < ear_threshold:
            status = "⚠️ DROWSY - WAKE UP! ⚠️"
            status_color = "red"
        else:
            status = "😊 AWAKE"
            status_color = "green"
    else:
        cv2.putText(frame, "NO FACE DETECTED", (w//2 - 100, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Left Eye EAR", f"{left_ear:.3f}")
    with col2:
        st.metric("Right Eye EAR", f"{right_ear:.3f}")
    with col3:
        st.metric("Average EAR", f"{avg_ear:.3f}")
    with col4:
        st.metric("Status", status)
    
    # Display status with color
    if status_color == "red":
        st.error(f"🚨 {status} 🚨")
    else:
        st.success(status)
    
    # Display processed image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Detection Output", use_container_width=True)
    
    # Alert sound (requires user interaction - browsers restrict auto-play)
    if avg_ear < ear_threshold and avg_ear > 0:
        st.warning("⚠️ ALERT: Driver appears drowsy! ⚠️")
        st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3", autoplay=True)
else:
    st.info("👆 Click 'Browse files' above to upload an image or use your camera")

st.markdown("---")
st.caption("Real-time driver drowsiness detection using MediaPipe Face Mesh")