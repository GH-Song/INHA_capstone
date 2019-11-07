## TOTAL_SUB대신 개인별 probablilty를 활용할 예정

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

reftime = pytime()
def getTime(s, referencetime = 0):
    ss = s / 1 - referencetime
    return ss

########################실행시 고려할 부분########################
# 가장 바깥에서, 딕셔너리를 통해, 각각의 이름에 대한 인스턴스 생성
# 분류 가능한 이름들
# 기준값 (전역변수)

TH_release = 1 # 100 %
rec_pause = 0

# 가장 바깥에서, 딕셔너리를 통해, 각각의 이름에 대한 인스턴스 생성
# 기준값
TH_of_confidence = 0.6
TH_of_Movement = 0.1

# 분류 가능한 이름들
names = ["Song_GH", "Kim_JW", "Choi_EH"]
names_detected = []

# 각 이름들로 찾을 수 있는 speak_utils의 인스턴스의 딕셔너리 생성
man = {name: speak_utils(name, TH_of_Movement) for name in names}

# 시간 동기화
First_time = getTime(pytime(), reftime)
for key in names:
    man[key].time1 = getTime(pytime(), reftime)
#####################################################################

# <editor-fold>
#region

# 파일 및 폴더 경로 지정

PATH_facedetection='face_detection_model'
PATH_recognizer = 'output/recognizer.pickle'
PATH_le = 'output/le.pickle'

# grab the indexes of the facial landmarks for the left and
outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(1.0)

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
            name = le.classes_[j]
            # 얼굴이 검출된 이름만 따로 저장 -> 이후 작업은 이 이름에 대해서만
            if name != "Unknown":
                names_detected.append(name)
                # facial landmark 추출해서 인스턴스 멤버에 저장
                # self.inmarks, self.outmarks, self.midmark 값이 부여됨
                man[name].landmark(gray, startX, startY, endX, endY)

                # 시간 미분 구하기 - 클래스 함수로

                # 미분 전에 미리 해줘야 할 것들
                detection_time = getTime(pytime(), reftime)
                # specific values, time2 update
                man[name].calculate_self(detection_time) 

                # 미분 함수 호출
                # time2 - time1에 대해, specific_value의 변화 계산
                # self.Mouth_movement에 반영
                # 계산시  time1 초기화
                man[name].differential(0.3, 2)

    # 직전의 검출 결과의 업데이트를 반영해서
    #print(TOTAL_SUB.items())
    # 만들어줘야 할 변수: starx-endy,
    if detections.shape[2] > 0:
        TOTAL_SUB_list = []
        for name in names_detected:
            TOTAL_SUB_list.append(man[name].TOTAL_SUB)
        probability_list = []
        if sum(TOTAL_SUB_list) >= 1:
            for name in names_detected:
                man[name].probability = man[name].TOTAL_SUB/sum(TOTAL_SUB_list)
        probability_list = [ man[name].probability for name in names_detected ]
        print("prob_list", sum(probability_list))
        print( getTime(pytime(), First_time) )
        print("rec_pause:", rec_pause)
        print( getTime(pytime(), First_time) > rec_pause )
        if getTime(pytime(), First_time) > rec_pause:
            if sum(probability_list) > 0:
            # 누가 가장 입을 많이 움직였는가 확인할 것
                for name in names_detected:
                    if man[name].probability >= max(probability_list):
                        # 현재 name을 가진 사람이 가장 말을 많이 했다면
                        # 화자 검출 성공
                        # 일단 초기화
                        print("speaker detected")
                        for key in names:
                            man[key].refresh("all") # 중요 변수 초기화
                        # 화자임을 표시
                        man[name].color_a = 50
                        man[name].color_b = 50
                        First_time = getTime(pytime())
                        rec_pause = 1
                        TH_release = 0.8
                        break
                    else:
                        man[name].refresh("color")
            else:
                # 아무도 말을 하지 않았으면
                for name in names:
                    man[name].refresh("color")
                TH_release = 1
        #####################출력값 지정##############################
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
    ###############################################################
    # update the FPS counter
    fps.update()
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
       break

# stop the timer and display FPS information
    fps.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
