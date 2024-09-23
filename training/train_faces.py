

import os
import face_recognition
import pickle

def train_faces():
    known_face_encodings = []
    known_face_names = []

    # Directory containing images
    path = "faces"

    print(f"Looking for images in directory: {path}")
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return

    # Loop through all files in the 'faces' directory
    for file_name in os.listdir(path):
        print(f"Processing file: {file_name}")
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            file_path = os.path.join(path, file_name)  # Combine directory path and file name
            print(f"File path: {file_path}")
            try:
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:  # Ensure that at least one face encoding was found
                    known_face_encodings.append(encoding[0])
                    known_face_names.append(os.path.splitext(file_name)[0])  # Name is the file name without extension
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Save encodings and names to a pickle file
    with open('face_encodings.pickle', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print("Training complete and face encodings saved.")

if __name__ == "__main__":
    train_faces()
