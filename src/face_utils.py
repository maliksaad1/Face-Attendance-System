import face_recognition
import cv2
import numpy as np
import streamlit as st
import time
from face_processor import FaceProcessor

class FaceUtils:
    def __init__(self):
        self.face_processor = FaceProcessor()
    
    def capture_face(self):
        """Capture face with liveness detection"""
        return self.face_processor.capture_face()
    
    def get_face_encoding(self, image):
        """Get face encoding from image"""
        return self.face_processor.get_face_encoding(image)
    
    def recognize_face(self, image, known_face_encodings, known_face_names):
        """Recognize faces in image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            face_names.append(name)
            
        return face_locations, face_names
    
    def draw_results(self, image, face_locations, face_names):
        """Draw bounding boxes and names on image"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        return image