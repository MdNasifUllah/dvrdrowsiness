#!/usr/bin/env python3
"""
Driver Drowsiness Detection System
Python version using OpenCV, MediaPipe, and Pygame
"""

import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import sys
from collections import deque
import threading

# Configuration parameters
CONFIG = {
    'EAR_THRESHOLD': 0.22,       # Threshold for eye closure (below this = closed)
    'FRAME_THRESHOLD': 30,       # Number of consecutive frames before alert
    'ALERT_SOUND_VOLUME': 0.7,   # Alert volume (0 to 1)
    'CAMERA_WIDTH': 640,         # Camera width
    'CAMERA_HEIGHT': 480,        # Camera height
    'ALERT_SOUND_FILE': 'alert.wav',  # Sound file (optional, creates beep if not found)
}

class DrowsinessDetector:
    def __init__(self):
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices (MediaPipe FaceMesh)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # State variables
        self.is_running = False
        self.closed_frames = 0
        self.alert_playing = False
        
        # Metrics
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.avg_ear = 0.0
        
        # Initialize pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Create a beep sound using numpy (fallback if no sound file)
        self.create_beep_sound()
        
    def create_beep_sound(self):
        """Create a beep sound programmatically"""
        try:
            # Try to load external sound file first
            self.alert_sound = pygame.mixer.Sound(CONFIG['ALERT_SOUND_FILE'])
        except:
            # Create a beep sound using pygame's sndarray
            sample_rate = 22050
            duration = 0.5
            frequency = 880
            
            frames = int(duration * sample_rate)
            arr = np.array([4096 * np.sin(2 * np.pi * frequency * x / sample_rate) for x in range(frames)])
            arr = arr.astype(np.int16)
            
            # Create stereo sound
            arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
            
            sound = pygame.sndarray.make_sound(arr)
            sound.set_volume(CONFIG['ALERT_SOUND_VOLUME'])
            self.alert_sound = sound
    
    def create_alert_sound2(self):
        """Create a different alert sound for no face detection"""
        sample_rate = 22050
        duration = 0.2
        frequency = 1760
        
        frames = int(duration * sample_rate)
        # Triangle wave approximation
        arr = np.array([2048 * (2 * abs(2 * (frequency * x / sample_rate - np.floor(0.5 + frequency * x / sample_rate))) - 1) 
                        for x in range(frames)])
        arr = arr.astype(np.int16)
        arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
        
        sound = pygame.sndarray.make_sound(arr)
        sound.set_volume(CONFIG['ALERT_SOUND_VOLUME'])
        return sound
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
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
    
    def play_alert_sound(self):
        """Play alert sound"""
        if self.alert_playing:
            return
        
        self.alert_playing = True
        
        def play():
            self.alert_sound.play()
            time.sleep(0.5)
            self.alert_playing = False
        
        threading.Thread(target=play, daemon=True).start()
    
    def play_no_face_alert(self):
        """Play alert when no face is detected"""
        if self.alert_playing:
            return
        
        self.alert_playing = True
        
        def play():
            sound = self.create_alert_sound2()
            sound.play()
            time.sleep(0.2)
            self.alert_playing = False
        
        threading.Thread(target=play, daemon=True).start()
    
    def draw_landmarks(self, frame, landmarks, h, w):
        """Draw face and eye landmarks on the frame"""
        # Draw all face landmarks (green dots)
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw eye landmarks (yellow dots)
        # Left eye
        for idx in self.LEFT_EYE_INDICES:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # Right eye
        for idx in self.RIGHT_EYE_INDICES:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        return frame
    
    def draw_status_panel(self, frame):
        """Draw status panel and metrics on frame"""
        h, w = frame.shape[:2]
        
        # Create overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text
        if self.closed_frames >= CONFIG['FRAME_THRESHOLD']:
            status = "⚠️ DROWSY - WAKE UP! ⚠️"
            status_color = (0, 0, 255)  # Red
        elif self.closed_frames >= CONFIG['FRAME_THRESHOLD'] * 0.5:
            status = "GETTING SLEEPY"
            status_color = (0, 165, 255)  # Orange
        elif self.closed_frames > 0:
            status = "BLINKING"
            status_color = (0, 255, 255)  # Yellow
        else:
            status = "AWAKE"
            status_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, status_color, 2, cv2.LINE_AA)
        
        # Metrics
        cv2.putText(frame, f"Left EAR: {self.left_ear:.3f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Right EAR: {self.right_ear:.3f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Avg EAR: {self.avg_ear:.3f}", (200, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Closed Frames: {self.closed_frames}", (200, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['CAMERA_WIDTH'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['CAMERA_HEIGHT'])
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        self.is_running = True
        print("Drowsiness Detection System Started")
        print("Press 'q' to quit")
        print("Press 'r' to reset")
        print("-" * 40)
        
        # For FPS calculation
        fps_counter = deque(maxlen=30)
        last_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get eye landmarks
                left_eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in self.LEFT_EYE_INDICES])
                right_eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in self.RIGHT_EYE_INDICES])
                
                # Calculate EAR
                self.left_ear = self.calculate_ear(left_eye_points)
                self.right_ear = self.calculate_ear(right_eye_points)
                self.avg_ear = (self.left_ear + self.right_ear) / 2
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, landmarks, h, w)
                
                # Drowsiness detection logic
                if self.avg_ear < CONFIG['EAR_THRESHOLD']:
                    self.closed_frames += 1
                    
                    if self.closed_frames >= CONFIG['FRAME_THRESHOLD']:
                        # Critical drowsiness
                        self.play_alert_sound()
                else:
                    # Eyes open - reset counter
                    if self.closed_frames >= CONFIG['FRAME_THRESHOLD']:
                        # Just woke up
                        print("Driver is awake now")
                    self.closed_frames = max(0, self.closed_frames - 1)
            else:
                # No face detected
                self.left_ear = 0.0
                self.right_ear = 0.0
                self.avg_ear = 0.0
                self.closed_frames = 0
                frame = self.draw_landmarks(frame, [], h, w)  # Clear landmarks
                cv2.putText(frame, "NO DRIVER DETECTED", (w//2 - 100, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                self.play_no_face_alert()
            
            # Draw status panel
            frame = self.draw_status_panel(frame)
            
            # Add title
            cv2.putText(frame, "Mil Dvr Drowsiness Detection Sys", (w//2 - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add footer
            cv2.putText(frame, "8 SIG BN || 24 INF DIV", (w - 150, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Calculate and display FPS
            current_time = time.time()
            fps_counter.append(1.0 / (current_time - last_time))
            last_time = current_time
            avg_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 80, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Show frame
            cv2.imshow('Driver Drowsiness Detection System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("Resetting counter...")
                self.closed_frames = 0
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("System stopped")

def main():
    print("=" * 50)
    print("DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 50)
    print(f"EAR Threshold: {CONFIG['EAR_THRESHOLD']}")
    print(f"Frame Threshold: {CONFIG['FRAME_THRESHOLD']}")
    print(f"Alert Volume: {CONFIG['ALERT_SOUND_VOLUME']}")
    print("=" * 50)
    
    detector = DrowsinessDetector()
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()