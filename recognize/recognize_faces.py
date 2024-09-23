import cv2
import face_recognition
import numpy as np
import pickle
import os
import mysql.connector
from datetime import datetime, timedelta

# Dictionary to store last marked time for each person
last_marked_time = {}

# Function to mark attendance
def mark_attendance(name):
    # Connect to MySQL database
    try:
        conn = mysql.connector.connect(
            host="localhost",  # Or the host of your MySQL server
            user="root",       # Your MySQL username
            password="Jesusislord7",  # Your MySQL password
            database="attendance_db"  # The database you created
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return

    cursor = conn.cursor()

    # Get the current datetime
    now = datetime.now()
    today_date = now.strftime('%Y-%m-%d')
    
    # Check if the person has already been marked today
    cursor.execute(
        "SELECT * FROM attendance WHERE name=%s AND DATE(date_time)=%s",
        (name, today_date)
    )
    result = cursor.fetchone()  # Fetch a single result

    # Fetch all remaining results (to avoid "Unread result found" error)
    cursor.fetchall()  # Ensure all results are fetched and cleared

    if result is None:  # No entry found for today
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

        # Insert the attendance data into the table
        sql = "INSERT INTO attendance (name, date_time) VALUES (%s, %s)"
        values = (name, dt_string)
        
        cursor.execute(sql, values)
        conn.commit()
        print(f"Attendance marked for {name} at {dt_string}")
    else:
        print(f"Attendance already marked for {name} today")

    # Close the cursor and connection
    cursor.close()
    conn.close()

# Function to recognize faces using webcam
def recognize_faces():
    # Load trained face encodings and names
    with open('training/face_encodings.pickle', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    # Start the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through faces in the frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Find best match for the face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
                # Check if sufficient time has passed since last attendance marking
                if name not in last_marked_time or datetime.now() - last_marked_time[name] > timedelta(minutes=5):
                    mark_attendance(name)
                    last_marked_time[name] = datetime.now()

            # Draw rectangle and name on the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        # Display video
        cv2.imshow('Video', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
