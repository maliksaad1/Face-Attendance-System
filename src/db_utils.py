from datetime import datetime
import numpy as np
from firebase_admin import db
import streamlit as st

class DatabaseUtils:
    def __init__(self):
        try:
            self.users_ref = db.reference('users')
            self.attendance_ref = db.reference('attendance')
        except Exception as e:
            st.error(f"Failed to initialize database references: {str(e)}")
    
    def save_user(self, name, face_encoding):
        """Save user data to Firebase"""
        try:
            user_data = {
                'name': name,
                'face_encoding': face_encoding.tolist(),  # Convert numpy array to list
                'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.users_ref.child(name.lower().replace(' ', '_')).set(user_data)
            return True
        except Exception as e:
            st.error(f"Failed to save user data: {str(e)}")
            return False
    
    def get_all_users(self):
        """Get all registered users"""
        try:
            users = self.users_ref.get()
            if not users:
                return [], []
            
            known_face_encodings = []
            known_face_names = []
            
            for user_id, user_data in users.items():
                known_face_encodings.append(np.array(user_data['face_encoding']))
                known_face_names.append(user_data['name'])
            
            return known_face_encodings, known_face_names
        except Exception as e:
            st.error(f"Failed to get users: {str(e)}")
            return [], []
    
    def mark_attendance(self, name):
        """Mark attendance for a user"""
        try:
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            time = now.strftime('%H:%M:%S')
            
            attendance_data = {
                'name': name,
                'time': time,
                'date': date
            }
            
            # Use date as parent node and time as child node to prevent duplicates
            self.attendance_ref.child(date).child(name.lower().replace(' ', '_')).set(attendance_data)
            return True
        except Exception as e:
            st.error(f"Failed to mark attendance: {str(e)}")
            return False
    
    def get_attendance_records(self, date=None):
        """Get attendance records for a specific date or all dates"""
        try:
            if date:
                records = self.attendance_ref.child(date).get()
                return records if records else {}
            
            return self.attendance_ref.get() or {}
        except Exception as e:
            st.error(f"Failed to get attendance records: {str(e)}")
            return {} 