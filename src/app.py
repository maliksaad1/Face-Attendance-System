import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
import face_recognition
from face_utils import FaceUtils
from db_utils import DatabaseUtils

# Initialize Firebase only if it hasn't been initialized yet
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://iot-monitoring-system-994de-default-rtdb.firebaseio.com/'
        })
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        st.info("Please make sure you have:")
        st.info("1. Created a Firebase project")
        st.info("2. Created a Realtime Database")
        st.info("3. Downloaded the service account key (firebase_key.json)")
        st.info("4. Updated the database URL in the code")

# Initialize utilities
face_utils = FaceUtils()
db_utils = DatabaseUtils()

# Page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide"
)

def register_face_page():
    st.subheader("Register New Face")
    
    # Initialize session state variables
    if 'registration_step' not in st.session_state:
        st.session_state.registration_step = 'input_name'
    if 'face_encoding' not in st.session_state:
        st.session_state.face_encoding = None
    
    # Step 1: Input Name
    if st.session_state.registration_step == 'input_name':
        name = st.text_input("Enter your name", key="register_name")
        if name and st.button("Start Registration", key="start_registration"):
            st.session_state.name = name
            st.session_state.registration_step = 'capture_face'
            st.rerun()
    
    # Step 2: Capture Face
    elif st.session_state.registration_step == 'capture_face':
        if st.session_state.face_encoding is None:
            face_encoding = face_utils.capture_face()
            if face_encoding is not None:
                if db_utils.save_user(st.session_state.name, face_encoding):
                    st.success(f"Face registered successfully for {st.session_state.name}!")
                else:
                    st.error("Failed to save user data")

def mark_attendance_page():
    st.subheader("Mark Attendance")
    
    if st.button("Mark Attendance", key="mark_attendance"):
        # Get registered users
        known_face_encodings, known_face_names = db_utils.get_all_users()
        if not known_face_encodings:
            st.warning("No registered users found. Please register some users first.")
            return
            
        # Capture and verify face
        face_encoding = face_utils.capture_face()
        
        if face_encoding is not None:
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
                # Mark attendance
                if db_utils.mark_attendance(name):
                    st.success(f"Attendance marked for {name}")
            else:
                st.error("Face not recognized. Please register first.")

def view_attendance_page():
    st.subheader("Attendance Analytics")
    
    # Get all records for analytics
    all_records = db_utils.get_attendance_records()
    
    if all_records:
        # Convert records to DataFrame for analysis
        data = []
        for date_records in all_records.values():
            if isinstance(date_records, dict):  # Handle single day records
                for record in date_records.values():
                    data.append({
                        'Name': record['name'],
                        'Time': record['time'],
                        'Date': record['date']
                    })
        
        import pandas as pd
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Attendance Records", len(df))
            
        with col2:
            unique_attendees = df['Name'].nunique()
            st.metric("Unique Attendees", unique_attendees)
        
        # Attendance Trends
        st.subheader("Daily Attendance Trends")
        daily_counts = df.groupby('Date').size().reset_index(name='Count')
        st.line_chart(daily_counts.set_index('Date'))
        
        # Attendance by Person
        st.subheader("Attendance by Person")
        person_counts = df['Name'].value_counts()
        st.bar_chart(person_counts)
        
        # Time Analysis
        st.subheader("Attendance Time Distribution")
        df['Hour'] = pd.to_datetime(df['Time'].astype(str)).dt.hour
        time_dist = df['Hour'].value_counts().sort_index()
        st.bar_chart(time_dist)
        
        # Recent Records
        st.subheader("Recent Attendance Records")
        recent_df = df.sort_values('Date', ascending=False).head(10)
        st.table(recent_df[['Date', 'Name', 'Time']].style.format({
            'Date': lambda x: x.strftime('%Y-%m-%d')
        }))
        
    else:
        st.info("No attendance records found in the database")

def main():
    st.title("Face Recognition Attendance System")
    
    # Sidebar navigation
    menu = ["Home", "Register Face", "Mark Attendance", "View Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Welcome to Face Recognition Attendance System")
        st.write("""
        This system allows you to:
        * Register your face using the camera
        * Mark attendance through face recognition
        * View attendance records
        """)
        
    elif choice == "Register Face":
        register_face_page()
            
    elif choice == "Mark Attendance":
        mark_attendance_page()
            
    elif choice == "View Attendance":
        view_attendance_page()

if __name__ == '__main__':
    main() 