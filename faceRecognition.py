import numpy as np
import cv2 as cv

har_cascade = cv.CascadeClassifier('C:/Users/shash/PycharmProjects/OpenCV/Advance/harcascadeFace.xml')

people = ['Anuv Jain', 'Modi Ji', 'Walter White']
#features = np.load('features.npy', allow_pickle=True)
#lavels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\shash\PycharmProjects\OpenCV\Advance\FaceRecognition\face.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

faces_rect = har_cascade.detectMultiScale(gray, 1.1, 10)

for(x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with accuracy of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('detected face', img)
cv.waitKey(0)
