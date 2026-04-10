import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import time
import base64
from io import BytesIO

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("### Mil Dvr Drowsiness Detection Sys")
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

@st.cache_resource
def load_drawing_utils():
    return mp.solutions.drawing_utils

face_mesh = load_face_mesh()
drawing_utils = load_drawing_utils()

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR)"""
    if len(eye_landmarks) != 6:
        return 0.0
    
    # Calculate vertical distances
    vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Calculate horizontal distance
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if horizontal == 0:
        return 0.0
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def process_image(image, ear_threshold):
    """Process single image for drowsiness detection"""
    # Convert PIL to numpy array
    frame = np.array(image)
    
    # Convert RGB to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.flip(frame_bgr, 1)
    h, w = frame_bgr.shape[:2]
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Default values
    left_ear = 0.0
    right_ear = 0.0
    avg_ear = 0.0
    status = "NO FACE DETECTED"
    status_color = "red"
    closed_frames = 0
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get eye landmarks
        left_eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE_INDICES])
        right_eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE_INDICES])
        
        # Calculate EAR
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2
        
        # Draw all face landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)
        
        # Highlight eyes in yellow
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 255), -1)
        
        # Draw eye contours
        left_eye_contour = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in LEFT_EYE_INDICES])
        right_eye_contour = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in RIGHT_EYE_INDICES])
        cv2.polylines(frame_bgr, [left_eye_contour], True, (0, 255, 255), 1)
        cv2.polylines(frame_bgr, [right_eye_contour], True, (0, 255, 255), 1)
        
        # Determine status
        if avg_ear < ear_threshold:
            status = "⚠️ DROWSY - WAKE UP! ⚠️"
            status_color = "red"
            closed_frames = 30  # Indicate drowsiness
        else:
            status = "😊 AWAKE"
            status_color = "green"
    else:
        cv2.putText(frame_bgr, "NO FACE DETECTED", (w//2 - 100, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add status overlay
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
    
    # Add text
    cv2.putText(frame_bgr, f"Status: {status.replace('⚠️', '').strip()}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Left EAR: {left_ear:.3f}", (20, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Right EAR: {right_ear:.3f}", (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Avg EAR: {avg_ear:.3f}", (180, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add title and footer
    cv2.putText(frame_bgr, "Mil Dvr Drowsiness Detection Sys", (w//2 - 150, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Convert back to RGB for display
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    return frame_rgb, left_ear, right_ear, avg_ear, status, status_color, closed_frames

# Import cv2 after setting page config (but we need it for the function)
try:
    import cv2
except ImportError:
    st.error("OpenCV is not installed. Please check requirements.txt")
    st.stop()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection Settings")
    ear_threshold = st.slider(
        "EAR Threshold (Lower = More Sensitive)", 
        min_value=0.15, 
        max_value=0.35, 
        value=0.22, 
        step=0.01,
        help="Eye Aspect Ratio below this value indicates closed eyes"
    )
    st.markdown("---")
    st.info("💡 **Tips for best results:**\n"
            "• Ensure good lighting\n"
            "• Face the camera directly\n"
            "• Remove glasses if possible\n"
            "• Keep face clearly visible")
    st.markdown("---")
    st.caption("8 SIG BN || 24 INF DIV")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Camera Input")
    camera_image = st.camera_input("Position your face in frame", key="camera")
    
    if camera_image is not None:
        with st.spinner("Processing image..."):
            image = Image.open(camera_image)
            processed_image, left_ear, right_ear, avg_ear, status, status_color, closed_frames = process_image(image, ear_threshold)
            
            # Display metrics
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            with col_metrics1:
                st.metric("Left Eye EAR", f"{left_ear:.3f}")
            with col_metrics2:
                st.metric("Right Eye EAR", f"{right_ear:.3f}")
            with col_metrics3:
                st.metric("Average EAR", f"{avg_ear:.3f}")
            with col_metrics4:
                st.metric("Drowsiness Score", f"{closed_frames}/30")
            
            # Display status
            if status_color == "red" and closed_frames >= 30:
                st.error(f"🚨🚨 {status} 🚨🚨")
                # Auto-play alert sound (HTML5 audio)
                alert_html = """
                <audio autoplay>
                    <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" type="audio/mpeg">
                </audio>
                """
                st.components.v1.html(alert_html, height=0)
            elif status_color == "red":
                st.warning(f"⚠️ {status} ⚠️")
            else:
                st.success(status)
            
            # Display processed image
            st.image(processed_image, caption="Detection Output", use_container_width=True)
    else:
        st.info("👆 Click 'Browse files' above to upload an image or use your camera")
        st.markdown("""
        ### How to use:
        1. Allow camera access when prompted
        2. Position your face in the frame
        3. The system will detect drowsiness based on eye closure
        4. Alert will trigger if eyes remain closed
        """)

with col2:
    st.subheader("📊 Real-time Information")
    st.markdown("""
    ### Understanding EAR (Eye Aspect Ratio)
    
    - **Normal (0.25-0.40)**: Eyes open, alert
    - **Blinking (0.20-0.25)**: Temporary closure
    - **Drowsy (<0.22)**: Eyes closing or closed
    
    ### Alert Levels
    
    | EAR Value | Status |
    |-----------|--------|
    | > 0.25 | 😊 Awake |
    | 0.22 - 0.25 | 😴 Blinking |
    | < 0.22 | ⚠️ Drowsy |
    
    ### System Info
    - Face detection: MediaPipe FaceMesh
    - Landmarks: 468 facial points
    - Eye tracking: 12 points (6 per eye)
    """)
    
    # Add a simple gauge for EAR
    if 'avg_ear' in locals() and avg_ear > 0:
        st.subheader("🎯 Current EAR Gauge")
        ear_percentage = min(100, (avg_ear / 0.4) * 100)
        st.progress(int(ear_percentage))
        st.caption(f"EAR Value: {avg_ear:.3f} (Higher is better)")

st.markdown("---")
st.caption("Driver Drowsiness Detection System | Real-time monitoring using AI | Keep your eyes on the road")