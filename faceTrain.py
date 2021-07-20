import cv2 as cv
import os
import numpy as np

people = ['Modi Ji', 'Anuv Jain', 'Walter White']
DIR = r'C:\Users\shash\PycharmProjects\OpenCV\Advance\FaceRecognition\Faces'

har_cascade = cv.CascadeClassifier('C:/Users/shash/PycharmProjects/OpenCV/Advance/harcascadeFace.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            img_path = os.path.join(path, image)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = har_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

            for(x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()
print('training done')

features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
