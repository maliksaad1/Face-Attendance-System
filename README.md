# Face Recognition Attendance System

A modern attendance system using face recognition, featuring real-time face detection, registration, and attendance marking through a camera feed.

## Features
- Face registration through camera
- Real-time face recognition for attendance
- Edge case handling (liveness detection, lighting conditions)
- Modern web interface using Streamlit
- Secure data storage using Firebase

## Setup Instructions

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Firebase:
   - Create a Firebase project
   - Download your Firebase service account key
   - Save it as `firebase_key.json` in the project root

4. Run the application:
   ```bash
   streamlit run src/app.py
   ```

## Project Structure
- `src/` - Contains the main application code
- `templates/` - HTML templates
- `static/` - Static files (CSS, JS) 