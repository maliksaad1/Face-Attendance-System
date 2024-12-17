import face_recognition
import cv2
import numpy as np
from collections import deque
import math
import streamlit as st
import time


class FaceProcessor:
    def __init__(self):
        # Load OpenCV's pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        # For head movement tracking
        self.position_history = deque(maxlen=30)  # Stores last 30 face positions
        self.circle_completed = False
        self.start_position = None
        self.movement_threshold = 20  # Minimum distance to consider movement
        self.circle_threshold = 0.7  # Circle completion threshold (0-1)
    
    def detect_face(self, image):
        """Detect faces in the image using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def get_face_center(self, face):
        """Calculate center point of detected face"""
        x, y, w, h = face
        return (x + w//2, y + h//2)
    
    def calculate_movement_angle(self, center):
        """Calculate angle of movement relative to the start position"""
        if self.start_position is None:
            return None
        
        dx = center[0] - self.start_position[0]
        dy = center[1] - self.start_position[1]
        angle = math.atan2(dy, dx)
        return math.degrees(angle) % 360
    
    def check_circle_completion(self):
        """Check if the face movement forms a circle"""
        if len(self.position_history) < 20:  # Need minimum points to check
            return False

        # Convert positions to numpy array for calculations
        points = np.array(list(self.position_history))
        
        # Calculate center of the movement
        center = np.mean(points, axis=0)
        
        # Calculate distances from center to each point
        distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
        mean_distance = np.mean(distances)
        
        # Check if points form a roughly circular pattern
        distance_variance = np.std(distances) / mean_distance
        
        # Calculate angular coverage
        angles = []
        for point in points:
            angle = math.atan2(point[1] - center[1], point[0] - center[0])
            angles.append(angle)
        
        # Sort angles and calculate coverage
        angles = sorted(angles)
        angle_diffs = np.diff(angles)
        total_angle = sum(angle_diffs)
        
        # Check if movement forms a near-complete circle
        circle_complete = (distance_variance < 0.2 and  # Consistent radius
                         abs(total_angle) > 5.0)  # Close to 2π radians
        
        return circle_complete
    
    def track_head_movement(self, face):
        """Track head movement and check for circular pattern"""
        if len(face) == 0:
            return False, None
        
        center = self.get_face_center(face[0])
        self.position_history.append(center)
        
        if self.start_position is None:
            self.start_position = center
        
        movement_status = "Keep moving your head in a circle"
        if self.check_circle_completion():
            self.circle_completed = True
            movement_status = "Circle completed!"
        
        return self.circle_completed, movement_status
    
    def draw_movement_guide(self, image):
        """Draw circular movement guide and tracking visualization"""
        h, w = image.shape[:2]
        center = (w//2, h//2)
        radius = min(w, h)//4
        
        # Draw guide circle
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        
        # Draw movement history
        if len(self.position_history) > 1:
            points = np.array(list(self.position_history), dtype=np.int32)
            for i in range(len(points)-1):
                cv2.line(image, tuple(points[i]), tuple(points[i+1]), 
                        (0, 0, 255), 2)
        
        return image
    
    def check_liveness(self, image):
        """Advanced liveness detection using head movement"""
        faces = self.detect_face(image)
        
        if len(faces) == 0:
            return False, "No face detected", None
        
        # Track head movement
        is_live, movement_status = self.track_head_movement(faces)
        
        # Draw movement visualization
        image_with_guide = self.draw_movement_guide(image.copy())
        
        return is_live, movement_status, image_with_guide
    
    def get_face_encoding(self, image):
        """Get face encoding from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            return face_encodings[0] if face_encodings else None
        return None
    
    def draw_face_rectangle(self, image, faces):
        """Draw rectangles around detected faces"""
        image_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image_copy
    
    def capture_face(self):
        """Capture and verify face"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open camera! Please check your camera connection.")
                return None
            
            # Wait a bit for camera to initialize
            time.sleep(1)
            
            placeholder = st.empty()
            status_text = st.empty()
            status_text.info("Looking for face... Please look at the camera")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        status_text.error("Failed to capture frame from camera")
                        break
                    
                    # Detect faces and check liveness
                    faces = self.detect_face(frame)
                    
                    if len(faces) > 0:
                        # Draw rectangles around detected faces
                        frame_with_faces = frame.copy()
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Show live feed with face detection
                        placeholder.image(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB))
                        
                        # Get face encoding
                        face_encoding = self.get_face_encoding(frame)
                        if face_encoding is not None:
                            status_text.success("Face detected and captured!")
                            return face_encoding
                    else:
                        # Show live feed without face detection
                        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        status_text.warning("No face detected - Please position your face in frame")
                    
                    # Add a small delay to reduce CPU usage
                    time.sleep(0.1)
                    
            except Exception as e:
                status_text.error(f"Error during face capture: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            return None
            
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
        
        return None