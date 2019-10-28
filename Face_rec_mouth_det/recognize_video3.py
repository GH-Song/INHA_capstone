# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import imutils
import pickle
import time
import cv2
import dlib
import os


class conversation:
    '''대화와 관련된 클래스'''
    # 인식 가능한 이름
    def __init__(self, names):
        self.names = names
        self.COUNTER = {}
        self.TOTAL_SUB = {}
        self.color_a = {}
        self.color_b = {}
        for key in self.names:
            self.COUNTER[key] = 0
            self.TOTAL_SUB[key] = 0
            self.color_a[key] = 255
            self.color_b[key] = 255
    def speakerfind():
        return
    
    def innermouth_aspect_ratio(self, mouth):
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        D = dist.euclidean(mouth[0], mouth[4])

        # compute the mouth aspect ratio
        self.imar = (A + B + C) / (3.0 * D)

        # return the mouth aspect ratio
        return self.imar    
    # 얼굴 박스 면적 대비 입 면적 비율?
    def face_area(self, startX, startY, endX, endY):
        X = dist.euclidean(startX,endX)
        Y = dist.euclidean(startY,endY) 
        return X*Y
    # 바깥 입술 면적 대비 벌린 면적 비율?
    def outtermouth_area(self, mouth):
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[3], mouth[9])
        D = dist.euclidean(mouth[4], mouth[8])
        
        X = dist.euclidean(mouth[0], mouth[6])
        Y = (B+C+D) / 3
        return X*Y
    
    def Innermouth_area(self, mouth):
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        X = dist.euclidean(mouth[0], mouth[4])
        # compute the mouth aspect ratio
        Y = (A + B + C) / (3.0)
        # return the mouth aspect ratio
        return X*Y

    def specific_area(self, Innermouth, Outtermouth):
        Inner_area = self.Innermouth_area(Innermouth)
        outter_area = self.outtermouth_area(Outtermouth)
        specific_area = (Inner_area* 10)/outter_area
        
        #return specific_area
        return outter_area

    def differential_area(self, del_area, del_time):
        return del_area/del_time

    def outtermouth_aspect_ratio(self, mouth):
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[1], mouth[11])
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[3], mouth[9])
        D = dist.euclidean(mouth[4], mouth[8])
        E = dist.euclidean(mouth[5], mouth[7])
        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        F = dist.euclidean(mouth[0], mouth[6])

        # compute the mouth aspect ratio
        omar = (A+B+C+D+E) / (5.0 * F)

        # return the mouth aspect ratio
        return omar    

    def outtermouth_distfactor(self, mouth):
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        self.A = dist.euclidean(mouth[1], mouth[11])
        self.B = dist.euclidean(mouth[2], mouth[10])
        self.C = dist.euclidean(mouth[3], mouth[9])
        self.D = dist.euclidean(mouth[4], mouth[8])
        self.E = dist.euclidean(mouth[5], mouth[7])
        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        self.F = dist.euclidean(mouth[0], mouth[6])
        return
    def distance_factor(self, Innermouth, Outtermouth):
        return

cvn = conversation(["Song_GH", "Kim_JW", "Choi_EH", "Unknown"])


# construct the argument parser and parse the arguments
TH_of_confidence = 0.3
threshold = 0.1
TH_of_Movement = 8
FRAMES = 1
# initialize the frame COUNTers and the total number of blinks
# 분류 가능한 이름들
KEYS = ["Song_GH", "Kim_JW", "Choi_EH", "Unknown"]

COUNTER = {}
TOTAL_SUB = {}
color_a = {}
color_b = {}
specific_value1 = {}
specific_value2 = {}
time1 = {}
time2 = {}
Mouth_movement = {}
face_area = {}

for key in KEYS:
    COUNTER[key] = 0
    TOTAL_SUB[key] = 0
    color_a[key] = 255
    color_b[key] = 255
    Mouth_movement[key] = 0
    specific_value1[key] = range(0,7)
    specific_value2[key] = range(0,7)
    time1[key] = 0
    time2[key] = 0  
    face_area[key] = 0

# check the time
def getCurrentTime(s):
    ss = s / 1
    return ss

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0], mouth[4])

    # compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)

    # return the mouth aspect ratio
    return mar

#region
# 파일 및 폴더 경로 지정
PATH_predictor = "shape_predictor_68_face_landmarks.dat"
PATH_facedetection='face_detection_model'
PATH_recognizer = 'output/recognizer.pickle'
PATH_le = 'output/le.pickle'

# grab the indexes of the facial landmarks for the left and
outermouth_start, outermouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
innermouth_start, innermouth_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# initialize dlib's face detector (HOG-based) and then create
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(PATH_predictor)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(1.0)

First_time = getCurrentTime(time.time())

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

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()
#endregion
for key in KEYS:
    time1[key] = getCurrentTime(time.time())
timevar = 0
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > TH_of_confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

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
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability

            rect = dlib.rectangle(left=int(startX), top=int(startY), right=int(endX), bottom=int(endY))
            
            face_area[name] = cvn.face_area(startX,startY,endX,endY)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            Innermouth = shape[innermouth_start:innermouth_end]
            Inner_mar = cvn.innermouth_aspect_ratio(Innermouth)
            Outtermouth = shape[outermouth_start:innermouth_end]
            
            time2[name] = getCurrentTime(time.time())
            del_time = time2[name] - time1[name]
            cvn.outtermouth_distfactor(Outtermouth)
            specific_value2[name] = [cvn.imar, cvn.B, cvn.C, cvn.D, cvn.F]
                        
                #specific_value2[name] = cvn.specific_area(Innermouth, Outtermouth)
            Mouth_movement[name] = ( abs(specific_value2[name][0] - specific_value1[name][0])*20 + 
                                        abs(specific_value2[name][1] - specific_value1[name][1]) +
                                        abs(specific_value2[name][2] - specific_value1[name][2]) +
                                        abs(specific_value2[name][3] - specific_value1[name][3]) +
                                        abs(specific_value2[name][4] - specific_value1[name][4])
                                        )/ (del_time*10) 
            time1[name] = time2[name]
            specific_value1[name] = specific_value2[name]
            #outter_mar = cvn.outtermouth_aspect_ratio(Outtermouth)
            #mar = (Inner_mar+outter_mar) / 2

            if Mouth_movement[name] > TH_of_Movement:
              TOTAL_SUB[name] += 1

            Last_time = getCurrentTime(time.time())
            print(Last_time)
            print(First_time)
        # 2초동안 들린 소리가 누구 것에 가까운지 판단
            if (Last_time - First_time) > 1.5:
            #if timevar > 
            # 이전 검사 이후 1초가 지났으면
             if sum(TOTAL_SUB.values()) >= 1:
                # 딕셔너리 내의 모든 값들에 대해
                if TOTAL_SUB[name] >= max(list(TOTAL_SUB.values())):
                    for key in KEYS:
                        color_a[key] = 255
                        color_b[key] = 255
                        TOTAL_SUB[key] = 0
                    color_a[name] = 50
                    color_b[name] = 50
                    First_time = getCurrentTime(time.time())
                else:
                    color_a[name] = 255
                    color_b[name] = 255
             else:
                for key in KEYS:
                    color_a[key] = 255
                    color_b[key] = 255
        ######################출력값 지정##############################
        #print(TOTAL_SUB.items())
        text = "{}: {:.2f}%".format(name, proba * 100)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        #print(startX, startY, endX, endY)
        t2 = "{:.2f}".format(Mouth_movement[name]*2)
        for (x, y) in shape[innermouth_start:innermouth_end]:
           cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in shape[outermouth_start:outermouth_end]:
           cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in shape[28:29]:
           cv2.rectangle(frame, (x - 160, y - 150), (x + 160, y - 70), (255, 255, 255), -1)
           cv2.rectangle(frame, (x - 160, y - 150), (x + 160, y - 70), (color_a[name], 255, color_b[name]), 3)
           cv2.putText(frame, text, (x-160, y-130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
           cv2.putText(frame, t2, (x-130, y-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        ###############################################################
    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    print(TOTAL_SUB.items())
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
       break

# stop the timer and display FPS information
    fps.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()