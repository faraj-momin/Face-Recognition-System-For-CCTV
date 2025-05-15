import cv2
import numpy as np
import pandas as pd
import os
import csv
import datetime
import time

# Ensure necessary directories exist
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Take images and save them for training
def TakeImages():
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists:
        with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for _ in reader1:
                serial += 1
        serial = (serial // 2)
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(['SERIAL NO.', 'ID', 'NAME'])
            serial = 1

#Taking Credentials of user
    Id = input("Enter Your ID: ")
    name = input("Enter Your Name: ")
    if name.isalpha() or ' ' in name:
        cam = cv2.VideoCapture(0)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('Taking Images', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum >= 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = f"Images Taken for ID : {Id}"
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([serial, Id, name])
        print(res)
    else:
        print("Enter Correct name")

# Train the face recognizer
def TrainImages():
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = cv2.imread(imagePath, 0)
            img_numpy = np.array(PIL_img, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[2])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(Id)
        return faceSamples, ids

    faces, ids = getImagesAndLabels('TrainingImage')
    recognizer.train(faces, np.array(ids))
    recognizer.write('TrainingImageLabel/Trainner.yml')
    print("Profile Saved Successfully")

# Track images and mark attendance
def TrackImages():
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name = df.loc[df['SERIAL NO.'] == Id]['NAME'].values[0]
                attendance = [str(Id), name, date, timeStamp]
                with open(f"Attendance/Attendance_{date}.csv", 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(attendance)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, name, (x + w, y), font, 1, (255, 255, 255), 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x + w, y), font, 1, (255, 255, 255), 2)
        cv2.imshow('Tracking Faces', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

# Main function to execute the system
def main():
    while True:
        print("\n1. Take Images\n2. Train Images\n3. Track Images\n4. Quit")
        choice = input("Enter your choice: ")
        if choice == '1':
            TakeImages()
        elif choice == '2':
            TrainImages()
        elif choice == '3':
            TrackImages()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()