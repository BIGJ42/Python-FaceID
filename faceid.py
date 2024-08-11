import cv2
import os
import numpy as np
import winsound
import json
from datetime import datetime
from deepface import DeepFace

# Directory to save recognized faces
recognized_faces_dir = "recognized_faces"
os.makedirs(recognized_faces_dir, exist_ok=True)

# JSON file to store face information
info_file = "recognized_faces_info.json"

# Define desired window size
window_width = 800
window_height = 600

# Initialize face_info dictionary
face_info = {}

# Load existing face information with error handling
try:
    with open(info_file, 'r') as f:
        file_content = f.read()
        if file_content.strip():  # Check if file is not empty
            face_info = json.loads(file_content)
except (json.JSONDecodeError, FileNotFoundError) as e:
    print(f"Error reading {info_file}: {e}")

# Initialize the face cascade
cascPath = "haarcascade_frontalface_default.xml"  # Provide a valid path if necessary
faceCascade = cv2.CascadeClassifier(cascPath)

# Initialize the video capture
video_capture = cv2.VideoCapture(1)

# Function to compare the current face with saved faces
def is_face_recognized(face_img):
    for filename in os.listdir(recognized_faces_dir):
        saved_face = cv2.imread(os.path.join(recognized_faces_dir, filename), cv2.IMREAD_GRAYSCALE)
        if saved_face is None:
            continue
        # Resize the new face and saved face to the same size
        saved_face_resized = cv2.resize(saved_face, (face_img.shape[1], face_img.shape[0]))
        difference = cv2.absdiff(saved_face_resized, face_img)
        if np.mean(difference) < 50:  # Adjust threshold based on testing
            return filename  # Return the filename if recognized
    return None

# Function to estimate age and gender
def estimate_age_gender(face_img):
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    result = DeepFace.analyze(face_img_rgb, actions=['age', 'gender'], enforce_detection=False)
    age = result[0]['age']
    gender = result[0]['gender']
    return age, gender

# Main loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        recognized_face_filename = is_face_recognized(face_img)
        if recognized_face_filename:
            status_text = "Recognized"
            color = (0, 255, 0)  # Green rectangle for recognized faces
            face_id = recognized_face_filename.split('.')[0]
            person_info = face_info.get(face_id, {})
            name = person_info.get("name", "Unknown")
            last_seen = person_info.get("last_seen", "Never")
            additional_info = f"Name: {name}, Last Seen: {last_seen}"

        else:
            face_id = f"face_{len(os.listdir(recognized_faces_dir)) + 1}"
            face_filename = f"{face_id}.jpg"
            cv2.imwrite(os.path.join(recognized_faces_dir, face_filename), face_img)

            person_info = {
                "name": f"Person {len(face_info) + 1}",
                "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            face_info[face_id] = person_info
            with open(info_file, 'w') as f:
                json.dump(face_info, f)

            status_text = "Not Recognized"
            color = (0, 0, 255)  # Red rectangle for new faces
            additional_info = "New face saved."
            winsound.Beep(1000, 500)

        age, gender = estimate_age_gender(face_img)
        additional_info = f"{additional_info}\nAge: {age}, Gender: {gender}"

        # Draw rectangle and display status
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, additional_info, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize the frame to the desired dimensions
    resized_frame = cv2.resize(frame, (window_width, window_height))

    # Display the resized frame
    cv2.imshow('Video', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
