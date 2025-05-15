Face Recognition Tool for CCTV

Description

This tool is a Python-based face recognition system designed to work with CCTV footage. It utilizes OpenCV for image processing and the LBPH face recognizer for face detection and recognition. The graphical user interface (GUI) is built using Tkinter, providing easy-to-use controls for capturing and tracking faces.

Features

Real-time face recognition using CCTV footage.

Image capture and profile saving functionality.

User authentication for data protection.

Attendance management with CSV export.

GUI-based interface for easy operation.

Prerequisites

Python 3.x

OpenCV

Tkinter

Numpy

Pillow

Pandas

Installation


Install the required libraries:

pip install -r requirements.txt

Run the application:

python main.py

Folder Structure

.
├── TrainingImage/             # Captured face images

├── TrainingImageLabel/        # Trained model data

├── StudentDetails/            # CSV files with user details

├── Attendance/                # Attendance records

├── haarcascade_frontalface_default.xml # Haar Cascade file for face detection

└── main.py                    # Main application file

How to Use

Run the application using the command:

python main.py

Use the 'Take Images' button to capture face data for a new user.      

Save the profile after capturing images.

Use 'Capture Face' to track and recognize faces.

Attendance records will be automatically saved in the 'Attendance' folder.

