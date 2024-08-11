# Face Recognition with Age and Gender Estimation

This project captures video from a webcam, detects faces, and performs face recognition, age, and gender estimation. It saves recognized faces and provides information about them. New faces are saved with placeholder names, and alerts are triggered for new faces.

## Features

- Face detection using OpenCV.
- Face recognition with saved face images.
- Age and gender estimation using DeepFace.
- Alerts for newly recognized faces.
- Display of face information on the video feed.

## Prerequisites

Ensure you have the following installed:

- Python 3.6 or later
- OpenCV
- NumPy
- TensorFlow (compatible version)
- DeepFace

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/BIGJ42/Python-FaceID.git
    cd Python-FaceID
    ```

2. Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have a valid Haar Cascade XML file for face detection (`haarcascade_frontalface_default.xml`).
2. Connect your webcam.
3. Run the script:

    ```bash
    faceid.py
    ```

4. The video feed will display faces detected and recognized, along with estimated age and gender. Press 'q' to quit the video feed.

## File Structure

- `faceis.py`: Main Python script for face detection and recognition.
- `recognized_faces/`: Directory where recognized face images are saved.
- `recognized_faces_info.json`: JSON file storing information about recognized faces.
- `requirements.txt`: List of dependencies.

## Dependencies

The `requirements.txt` file includes:

```txt
opencv-python==4.8.0.76
numpy==1.24.2
tensorflow==2.10.0
deepface==0.0.80
