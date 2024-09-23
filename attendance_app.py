from flask import Flask, render_template, Response, redirect, url_for
import cv2
import face_recognition
import numpy as np
import pickle
import mysql.connector
import pandas as pd
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Global Variables
video_capture = cv2.VideoCapture(0)  # Start the webcam
last_marked_time = {}

# Load trained face encodings and names
with open('training/face_encodings.pickle', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Function to mark attendance
def mark_attendance(name):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Jesusislord7",
            database="attendance_db"
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return

    cursor = conn.cursor()
    now = datetime.now()
    today_date = now.strftime('%Y-%m-%d')

    # Check if the person has already been marked today
    cursor.execute(
        "SELECT * FROM attendance WHERE name=%s AND DATE(date_time)=%s",
        (name, today_date)
    )
    result = cursor.fetchone()

    if result is None:
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S'
        sql = "INSERT INTO attendance (name, date_time) VALUES (%s, %s)"
        values = (name, dt_string)
        cursor.execute(sql, values)
        conn.commit()

        # Save the data to Excel
        attendance_data = pd.DataFrame({'Name': [name], 'Date_Time': [dt_string]})
        file_path = 'attendance.xlsx'

        if os.path.exists(file_path):
            attendance_data.to_excel(file_path, mode='a', header=False, index=False)
        else:
            attendance_data.to_excel(file_path, index=False)

        print(f"Attendance marked for {name} at {dt_string}")
    else:
        print(f"Attendance already marked for {name} today")

    cursor.close()
    conn.close()

# Function to recognize faces
def recognize_faces():
    global video_capture
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name not in last_marked_time or datetime.now() - last_marked_time[name] > timedelta(minutes=5):
                    mark_attendance(name)
                    last_marked_time[name] = datetime.now()

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to display the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Home page template with "Recognize Faces" button

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(recognize_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
