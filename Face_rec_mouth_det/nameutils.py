# import the necessary packages
from imutils.video import FileVideoStream, VideoStream, FPS
from imutils import face_utils
import numpy as np
import imutils
import pickle
import time
import cv2
import dlib
import os

# 파일 및 폴더 경로 지정
PATH_facedetection='face_detection_model'
PATH_recognizer = 'output/recognizer.pickle'
PATH_le = 'output/le.pickle'

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([PATH_facedetection, "deploy.prototxt"])
modelPath = os.path.sep.join([PATH_facedetection, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(PATH_recognizer, "rb").read())
le = pickle.loads(open(PATH_le, "rb").read())


class face_name_utils:
    def read_frame(self, frame):
        # dimensions
        (self.h, self.w) = frame.shape[:2]
        # construct a blob from the image
        self.imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    def detect_face(self):
        detector.setInput(self.imageBlob)
        self.detections = detector.forward()
        return self.detections

    def predict_name(self, face_index, frame, TH = 0.6):
        confidence = self.detections[0, 0, face_index, 2]
        # filter out weak detections
        if confidence > TH:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = self.detections[0, 0, face_index, 3:7] * np.array([self.w, self.h, self.w, self.h])
            (self.startX, self.startY, self.endX, self.endY) = box.astype("int")
            # call by reference
            face_box_index = [int(self.startX), int(self.startY), int(self.endX), int(self.endY)]
            # extract the face ROI
            face = frame[self.startY:self.endY, self.startX:self.endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                return "", [0,0,0,0]
            else:
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]

                # 이름이 부여되는 시점
                self.name = le.classes_[j]
                return self.name, face_box_index
        else:
            return "", [0,0,0,0]
