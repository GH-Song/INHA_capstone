#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
import speakutils
from speakutils import speak_utils
from time import time as pytime
import pandas as pd
from collections import OrderedDict

from datetime import datetime

# In[2]:

def getTime(s, referencetime = 0):
    ss = s / 1 - referencetime
    return np.float64(ss)
########################실행시 고려할 부분########################
# 가장 바깥에서, 딕셔너리를 통해, 각각의 이름에 대한 인스턴스 생성
# 분류 가능한 이름들
names = ["Song_GH"]
names_detected = []

# 각 이름들로 찾을 수 있는 speak_utils의 인스턴스의 딕셔너리 생성
size_of_buffer = 200

# 기준값
TH_of_confidence = 0.6
TH_of_Movement = 0.6
FRAMES = 1
rec_time_limit = 0 # 1회 측정시간 부여 시 사용
datalabel = ""

# 시간 동기화
reftime = getTime(pytime())
First_time = getTime(pytime())
####################################################################


# In[4]:

# <editor-fold>
#region

# 파일 및 폴더 경로 지정
PATH_facedetection='face_detection_model'
PATH_recognizer = 'output/recognizer.pickle'
PATH_le = 'output/le.pickle'

# grab the indexes of the facial landmarks for the left and
outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

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
# </editor-fold>


# In[5]:

# loop over frames from the video file stream
while True:
    # wait for key in terminal
    key = input("press 's' for speaking, 'n' for nonspeaking, 'q' for quit: \n")
    # if the `q` key was pressed, break from the loop
    if key == "q":
        print("program finished")
        break
    elif key == "s":
        # 앞으로 20초간 인식하도록 셋팅
        First_time = getTime(pytime())
        rec_time_limit = np.float64(10.0)
        print("rec_time_limit:", rec_time_limit)
        reftime = getTime(pytime())
        datalabel = "speaking"
        man = {name: speak_utils(name, buffersize = size_of_buffer) for name in names}
        for key in names:
            man[key].time1 = getTime(pytime(), reftime)
    elif key == "n":
        # 앞으로 20초간 인식하도록 셋팅
        First_time = getTime(pytime())
        rec_time_limit = np.float64(10.0)
        reftime = getTime(pytime())
        datalabel = "nonspeaking"
        man = {name: speak_utils(name, buffersize = size_of_buffer) for name in names}
        for key in names:
            man[key].time1 = getTime(pytime(), reftime)

    while getTime(pytime(), First_time) < rec_time_limit:
        print("del_time:", getTime(pytime(), First_time))
        # <editor-fold>
        # grab the frame from the threaded video stream
        frame = vs.read()
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (h, w) = frame.shape[:2]

        # while문에 대한 man.time1의 시간 동기화
        ######문제발견: timeset, testloop 등이 개인별로 있어야 함

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # 검출된 얼굴들에 대한 반복
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

                # 이름이 부여되는 시점
                if le.classes_[j] in names:
                    name = le.classes_[j]
                else:
                    continue
                # 얼굴이 검출된 이름만 따로 저장 -> 이후 작업은 이 이름에 대해서만
                names_detected.append(name)
                # facial landmark 추출해서 인스턴스 멤버에 저장
                # self.inmarks, self.outmarks, self.midmark 값이 부여됨
                man[name].landmark(gray, startX, startY, endX, endY)

                # 미분 전에 미리 해줘야 할 것들=
                detection_time = getTime(pytime(), reftime)
                man[name].calculate_self(detection_time) # 몇 번째 반복중인지 전달
                # 미분 함수 호출
                #man[name].differential(100, 1) # loop를 올리기만 하는 역할
                man[name].loop += 1
        #### detection loop나오기 (while문과 동일 위치)
        ######################출력값 지정##############################
        #print(TOTAL_SUB.items())
        # 만들어줘야 할 변수: starx-endy,
        if detections.shape[2] > 0:
            for name in names_detected:
                # 이름 출력 텍스트
                text = "{}: {:.2f}%".format(name, proba * 100)
                # Mouth_movement 출력 텍스트
                t2 = "{:.4f}".format(man[name].Mouth_movement)
                # 얼굴 테두리 사각형
                cv2.rectangle(frame, (man[name].sx, man[name].sy), (man[name].ex, man[name].ey), (0, 0, 255), 2)
                # 입술 주위 점
                for (x, y) in man[name].Inmarks:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                for (x, y) in man[name].Outmarks:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # 말풍선
                x, y = man[name].Midmark[0,0], man[name].Midmark[0,1]
                cv2.rectangle(frame, (x - 160, y - 150), (x + 160, y - 70), (255, 255, 255), -1)
                cv2.rectangle(frame, (x - 160, y - 150), (x + 160, y - 70), (man[name].color_a, 255, man[name].color_b), 3)
                cv2.putText(frame, text, (x-160, y-130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, t2, (x-130, y-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            names_detected = []
        # show the frame
        fps.update()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
           break
        # </editor-fold>
    else:
        cv2.destroyAllWindows()
        First_time = getTime(pytime())
        # First_time = Last_time
        rec_time_limit = 0
        name = "Song_GH"

        # if man[name].loop > 50:
        recorded_data = pd.DataFrame(
            OrderedDict([
                ('Name', np.full((size_of_buffer, 1), man[name].name).flatten()),
                ('Label', np.full((size_of_buffer, 1), datalabel).flatten()),
                ('system_time', np.full((size_of_buffer, 1), datetime.now).flatten()),
                ('record_time', man[name].specific_values[:,0].flatten()),
                ('InnerMAR', man[name].specific_values[:,1].flatten()),
                ('outter_width', man[name].specific_values[:,2].flatten())
            ])
        )
        print("record finish\n#########################\n", recorded_data)
        # recorded_data.to_pickle("output/recorded_mouth_data.pickle")
        if input("이번 데이터를 저장하시겠습니까? (y/n) ") == "y":
            print("/output 파일 목록:\n ", os.listdir(os.getcwd()+"/output") )
            fname = input("파일 이름을 입력하세요: ") + ".pickle"
            fpath = os.path.join("output", fname )
            try:
                metadata = pd.read_pickle(fpath)
                print("[INFO] Read DataFrame from pickle...")
                metadata = metadata.append(recorded_data)
                # metadata.to_pickle("output/recorded_mouth_data" + str(datetime.now)+ ".pickle")
                metadata.to_pickle(fpath)
                print("[INFO] Data is saved in pickle...")
            except:
                print("[INFO] 기존 파일이 존재하지 않아, 새로 생성합니다")
                recorded_data.to_pickle(fpath)
            # output/recorded_mouth_data.pickle'

            # 데이터 인덱스를 저장 날짜로 할 수 있도록 할 것
    ###############################################################

# stop the timer and display FPS information
    fps.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


# In[ ]:
