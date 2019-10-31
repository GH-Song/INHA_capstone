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

# my class
class speak_utils:
    '''대화와 관련된 클래스'''
    def __init__(self, name):
        self.name = name
        self.TOTAL_SUB = 0
        self.color_a = 255
        self.color_b = 255
        self.specific_value1 = range(0,7)
        self.specific_value2 = range(0,7)
        self.specific_values = np.zeros((100,7), dtype = np.float64)
        self.time1 = 0;    self.time2 = 0
        self.Mouth_movement = 0
        self.sx = 0;    self.sy = 0
        self.ex = 0;    self.ey = 0
        self.Inmarks = []
        self.Outmarks = []
        self.Midmark = []
        self.timeset = True
        self.loop = 0

    def landmark(self, gray, sx=0, sy=0, ex=0, ey=0):
        '''검출된 얼굴 좌표들을 저장'''
        # grab the indexes of the facial landmarks for the left and
        self.sx = int(sx);    self.sy = int(sy)
        self.ex = int(ex);    self.ey = int(ey)
        outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

        rect = dlib.rectangle(left=self.sx, top=self.sy, right=self.ex, bottom=self.ey)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        self.Inmarks = shape[inmark_start:inmark_end]
        self.Outmarks = shape[outmark_start: outmark_end]
        self.Midmark = shape[28:29]
        return

    def speakerfind():
        return

    def innermouth_aspect_ratio(self):
        '''입이 벌어진 정도를 확인하는 함수'''
        mouth = self.Inmarks
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

    def face_area(self):
        # 얼굴 면적 계산하는 함수

        X = dist.euclidean(self.sx, self.ex)
        Y = dist.euclidean(self.sy, self.ey)

        self.face_area = X*Y
        return self.face_area

    # 바깥 입술 면적 대비 벌린 면적 비율?
    def outtermouth_area(self):
        mouth = self.Outmarks
        # 바깥 입술의 면적을 계산하는 함수
        # 세로 선 중 3개의 y축 길이
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[3], mouth[9])
        D = dist.euclidean(mouth[4], mouth[8])

        X = dist.euclidean(mouth[0], mouth[6])
        Y = (B+C+D) / 3
        self.outtermouth_area = X*Y
        return self.outtermouth_area

    def Innermouth_area(self):
        mouth = self.inmarks
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

    def outtermouth_aspect_ratio(self):
        mouth = self.Outmarks
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

    def outtermouth_distfactor(self):
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        mouth = self.Outmarks
        self.OA = dist.euclidean(mouth[1], mouth[11])
        self.OB = dist.euclidean(mouth[2], mouth[10])
        self.OC = dist.euclidean(mouth[3], mouth[9])
        self.OD = dist.euclidean(mouth[4], mouth[8])
        self.OE = dist.euclidean(mouth[5], mouth[7])
        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        self.OF = dist.euclidean(mouth[0], mouth[6])
        return

    def distance_factor(self, Innermouth, Outtermouth):
        return

    def update_specific_value(self, option):
        if option == 1:
            self.specific_value1 = [self.imar, self.OB, self.OC, self.OD, self.OF]
        else:
            self.specific_value2 = [self.imar, self.OB, self.OC, self.OD, self.OF]

    def print_specific_values(self):
        print(self.name, "의 값:\n", self.specific_values[:5,:6])

    def update_specific_values(self, row=0):
        self.specific_values[row,:6] = [self.time2, self.imar, self.OB, self.OC, self.OD, self.OF]

    def difference_of_specific_values(self):
        # 행 1-0 2-1 3-2...
        difference = list(range(100))
        for i in range(self.loop + 1):
            difference[i] = self.specific_values[i+1, :6] - self.specific_values[i, :6]
        return difference
        # difference의 한 원소는 1행 6열 numpy

    def specific_values_cal(self):
        np.sum(self.specific_values[1,:5])
        array1 = self.specific_values[1,:5]
    def differential(self, timeref, option):
        '''Mouth_movement 값을 계산, 계산시 True반환'''
        if option == 1:
            # 계산방식 1
            del_time = self.time2 - self.time1
            print(del_time)
            if del_time > timeref:
                # 각 specific value에 대해, 가중치를 고려해서 입 움직임의 정도를 수치화
                self.Mouth_movement = (
                    abs(self.specific_value2[0] - self.specific_value1[0])*30 +
                    abs(self.specific_value2[1] - self.specific_value1[1]) +
                    abs(self.specific_value2[2] - self.specific_value1[2]) +
                    abs(self.specific_value2[3] - self.specific_value1[3]) +
                    abs(self.specific_value2[4] - self.specific_value1[4])*2
                )/ (del_time*110)
                self.time1 = self.time2
                self.specific_value1 = self.specific_value2
                return True
            else:
                return False
        elif option == 2:
            # 계산방식 2
            del_time = self.time2 - self.time1
            if del_time > timeref:
                del_values = self.difference_of_specific_values()
                # del_time 은 del_values[0][0]
                # 각 specific value에 대해, 가중치를 고려해서 입 움직임의 정도를 수치화
                self.Mouth_movement = 0
                for i in range(self.loop):
                    self.Mouth_movement += (
                        abs(del_values[i][1])*30 +
                        abs(del_values[i][2])+
                        abs(del_values[i][3])+
                        abs(del_values[i][4])+
                        abs(del_values[i][5])*2
                    )/ (del_values[i][0]*110)

                self.time1 = self.time2
                self.specific_value1 = self.specific_value2
                return True # 이것이 반환되면 self.loop초기화
            return False

    def calculate_self(self):
        # self.face_area()
        self.innermouth_aspect_ratio() # self.imar
        self.outtermouth_distfactor() # self.A,B,C..
        self.update_specific_value(2) # 위의 계산을 time2의 specific value에 반영
        self.update_specific_values(self.loop)
        self.print_specific_values()
        return

    def refresh(self, option):
        if option == "all":
            self.TOTAL_SUB = 0
            self.color_a = 255
            self.color_b = 255
        elif option == "color":
            self.color_a = 255
            self.color_b = 255
        elif option == "TOTAL_SUB" :
            self.TOTAL_SUB = 0

def getTime(s):
    ss = s / 1
    return ss

########################실행시 고려할 부분########################
# 가장 바깥에서, 딕셔너리를 통해, 각각의 이름에 대한 인스턴스 생성
# 분류 가능한 이름들
names = ["Song_GH", "Kim_JW", "Choi_EH", "Unknown"]
names_detected = []

# 각 이름들로 찾을 수 있는 speak_utils의 인스턴스의 딕셔너리 생성
man = {name: speak_utils(name) for name in names}

# {"Song_GH":inst1 , "Kim_JW":inst2, "Choi_EH":inst3, "Unknown":inst4}
# Man["Song_GH"] == inst1
# Man[name].COUNTER

# 기준값
TH_of_confidence = 0.6
#threshold = 0.1
TH_of_Movement = 0.9
FRAMES = 1

# 시간 동기화
First_time = getTime(time.time())
for key in names:
    man[key].time1 = getTime(time.time())
#####################################################################

# <editor-fold>
#region

# 파일 및 폴더 경로 지정
PATH_predictor = "shape_predictor_68_face_landmarks.dat"
PATH_facedetection='face_detection_model'
PATH_recognizer = 'output/recognizer.pickle'
PATH_le = 'output/le.pickle'

# grab the indexes of the facial landmarks for the left and
outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# initialize dlib's face detector (HOG-based) and then create
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(PATH_predictor)

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
            name = le.classes_[j]
            # 얼굴이 검출된 이름만 따로 저장 -> 이후 작업은 이 이름에 대해서만
            names_detected.append(name)
            # facial landmark 추출해서 인스턴스 멤버에 저장
            # self.inmarks, self.outmarks, self.midmark 값이 부여됨
            man[name].landmark(gray, startX, startY, endX, endY)

            # 시간 미분 구하기 - 클래스 함수로

            # 미분 전에 미리 해줘야 할 것들
            man[name].time2 = getTime(time.time()) # 시간 바깥 while문에서, 해당 for문으로 들어오면서,
            man[name].calculate_self() # 몇 번째 반복중인지 전달
            print(man[name].loop)

            # 미분 함수 호출
            # time2 - time1에 대해, specific_value의 변화 계산
            # self.Mouth_movement에 반영
            # 이 부분에 if loopnumber조건 추가 가능
            # 계산시 True가 되어 while문에서 time1 초기화

            man[name].timeset = man[name].differential(0.2, 2) # time2 - time1이 0.01보다 큰지 고려해서 실행됨
            ''' 1초 기준, 혼자있을 때, timeloop는 0 - 9 까지 증가
                0.5 혼자 4까지
                timeloop reset될 떄마다 numpy도 리셋해야함
            '''
            # 역치값과 비교
            if man[name].Mouth_movement > TH_of_Movement:
                man[name].TOTAL_SUB += 1*int(man[name].Mouth_movement/TH_of_Movement)

    #### detection loop나오기 (while문과 동일 위치)
    for name in names_detected:
        # 검출 안된 이름이 계속 쌓이는 문제
        if man[name].timeset == True:
            man[name].time1 = getTime(time.time())
            man[name].timeset = False
            man[name].loop = 0
        else:
            man[name].loop += 1

    # 직전의 검출 결과의 업데이트를 반영해서
    # 이름 기준으로 반복
    Last_time = getTime(time.time())
    # 2초동안 들린 소리가 누구 것에 가까운지 판단
    if (Last_time - First_time) > 1:
        ''' 이 시간을 초기화 하기 전에, 말을 한 사람에게 추가적인 시간을 더 주자.
            혹은 이 시간 자체를 없애고, 이전에 말을 한 사람이 검출된 시점을 기준으로 하자'''
        # 이전 검사 완료 이후 x초가 지났으면
        TOTAL_SUB_list = []
        # 모든 사람 인스턴스가 가진 self.TOTAL_SUB값을 리스트로 추출
        for key in names_detected:
            TOTAL_SUB_list.append(man[key].TOTAL_SUB)
        # 화자 검출
        if sum(TOTAL_SUB_list) >= 1:
            # 누군가 한번 이상 말을 했으면
            for name in names_detected:
            # 누가 가장 입을 많이 움직였는가 확인할 것
                if man[name].TOTAL_SUB >= max(TOTAL_SUB_list):
                    # 현재 name을 가진 사람이 가장 말을 많이 했다면
                    # 화자 검출 성공
                    # 일단 초기화
                    for key in names:
                        man[key].refresh("all") # 중요 변수 초기화
                    # 화자임을 표시
                    man[name].color_a = 50
                    man[name].color_b = 50
                    First_time = getTime(time.time())
                else:
                    man[name].refresh("color")
        else:
            # 아무도 말을 하지 않았으면
            for name in names:
                man[name].refresh("color")
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
            # 터미널 출력
            print(Last_time,'/', name, "'s TOTAL:", man[name].Mouth_movement)
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
