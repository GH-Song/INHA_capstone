# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import pickle
import time
import cv2
import dlib
import os

# face landmark 추출에 필요한 predictor모듈 기능 활성화
PATH_predictor = "shape_predictor_68_face_landmarks.dat"
print("[INFO] loading facial landmark predictor in speakutils...")
predictor = dlib.shape_predictor(PATH_predictor)

outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# my class
class speak_utils:
    '''대화 중인 사람의 얼굴에서 관측되는 특징값을 저장하는 변수들과
    분석하는 함수를 가진 클래스'''
    # 생성자
    def __init__(self, name, TH=0, buffersize = 20):
        self.TH_of_Movement = TH
        self.name = name
        self.buffersize = buffersize
        self.TOTAL_SUB = 0
        self.color_a = 255; self.color_b = 255
        self.specific_values = np.zeros((self.buffersize,7), dtype = np.float64)
        self.time1 = 0; self.time2 = 0
        self.Mouth_movement = 0
        self.sx = 0; self.sy = 0; self.ex = 0; self.ey = 0
        self.Inmarks = []; self.Outmarks = []; self.Midmark = []
        self.loop = 0
        self.probability = 0
        self.face_area = 0
        self.buf_top = 0
        self.sayingwords = ""
        print(self.name+"'s class is created")

    # 검출된 얼굴 세부 부위 좌표를 저장
    def landmark(self, gray, sx, sy, ex, ey):
        ''' 사용법: 이미 얼굴 좌표가 검출된 후 사용
        .landmark( frame, 왼쪽상단 x, 왼쪽 상단 y, 오른쪽 하단x, 오른쪽 하단 y )'''
        # 인스턴스 변수에 저장
        self.sx = sx; self.sy = sy; self.ex = ex; self.ey = ey
        # 얼굴 면적 계산
        X = self.sx - self.ex
        Y = self.sy - self.ey
        self.face_area = abs(X*Y)

        # 얼굴 세부 좌표 예측
        rect = dlib.rectangle(left=self.sx, top=self.sy, right=self.ex, bottom=self.ey)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 입술 안쪽 좌표
        self.Inmarks = shape[inmark_start:inmark_end]
        # 입술 바깥 좌표
        self.Outmarks = shape[outmark_start: outmark_end]
        # 얼굴 중앙 좌표
        self.Midmark = shape[28:29]
        return

    # 입이 벌어진 정도를 계산하는 함수
    def innermouth_aspect_ratio(self):
        '''매개변수 없이 스스로 계산'''
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

    # 바깥 입술 면적 계산
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

    def print_specific_values(self, data_start = 0, data_size = 5):
        data_end = data_start + data_size
        print(self.name, "의 값:\n", self.specific_values[data_start:data_end,:6])

    def update_specific_values(self, row=0):
        self.specific_values[row,:6] = [self.time2, self.imar, self.OB, self.OC, self.OD, self.OF]

    # 미분 수행, Mouth_movement값을 계산
    def differential(self, timeref, option, disp= False):
        ''' timeref: 미분을 수행할 시간 간격, option: 계산방식'''
        if option == 1:
            if (self.time2 - self.time1) > timeref and self.loop > 0:
                loop = self.buf_top + self.loop
                max_index = loop + 1

                # 시간 변화량 계산
                del_t = np.diff(self.specific_values[:max_index,0]).reshape(loop,1) # 열 끼리 뺄셈

                # 값 변화량 계산
                del_values = np.diff(self.specific_values[:max_index,1:6], axis = 0) # 열 끼리 뺄셈

                # 시간에 따른 값 변화율 계산
                dev_values = del_values/del_t
                total_dev = np.sum(abs(dev_values), axis=0).flatten() # 열 끼리 덧셈

                # 시간은 기준보다 지났으며, 측정횟수는 2회 이상인가
                print("---------", self.name, "의 미분결과--------")
                print("측정 누적횟수:", self.loop)
                print("시간변화:", del_t)
                print("값 변화:", del_values)

                # Mouth_movement 계산
                self.Mouth_movement = np.sum(total_dev)*100/self.face_area/max_index
                self.Mouth_movement += self.specific_values[:max_index,1].mean()

                # 시간 동기화
                self.time1 = self.time2
                # 버퍼 초기화
                #self.specific_values = np.zeros((self.buffersize,7), dtype = np.float64)
                # 반복 횟수를 나타내는 값 초기화
                self.loop = 0
            else:
                # 측정이 반복됨을 기록
                self.loop += 1

            # 역치값과 비교
            if self.Mouth_movement > self.TH_of_Movement:
                self.TOTAL_SUB += 1*int(self.Mouth_movement/self.TH_of_Movement)

        elif option == 2:
            # 계산방식 2
            if (self.time2 - self.time1) > timeref and self.loop > 0:
                # 시간은 기준보다 지났으며, 측정횟수는 2회 이상인가
                loop = self.loop
                max_index = loop + 1

                # 시간 변화량 계산
                del_t = np.diff(self.specific_values[:max_index,0]).reshape(loop,1) # 열 끼리 뺄셈

                # 값 변화량 계산
                del_values = np.diff(self.specific_values[:max_index,1:6], axis = 0) # 열 끼리 뺄셈

                # 시간에 따른 값 변화율 계산
                dev_values = del_values/del_t
                total_dev = np.sum(abs(dev_values), axis=0).flatten() # 열 끼리 덧셈

                # Mouth_movement 계산
                self.Mouth_movement = np.sum(total_dev)*100/self.face_area/max_index

                self.Mouth_movement += self.specific_values[:max_index,1].mean()

                if disp == True:
                    print("---------", self.name, "의 미분결과--------")
                    print("측정 누적횟수:", self.loop)
                    print("시간변화:", del_t)
                    print("값 변화:", del_values)
                    print("dev:", self.Mouth_movement/ (max_index+1))
                    print("mean imar:", self.specific_values[:max_index,1].mean())

                # 시간 동기화
                self.time1 = self.time2
                # 버퍼 초기화
                self.specific_values = np.zeros((self.buffersize,7), dtype = np.float64)
                # 반복 횟수를 나타내는 값 초기화
                self.loop = 0
            else:
                # 측정이 반복됨을 기록
                self.loop += 1

            # 역치값과 비교
            if self.Mouth_movement > self.TH_of_Movement:
                self.TOTAL_SUB += 1*int(self.Mouth_movement/self.TH_of_Movement)
                if disp == True:
                    print("Upper than threshold")
                    print("------------------------------------")
            else:
                if disp == True:
                    print("Lower than threshold")
                    print("------------------------------------")
        else:
            return

    # imar, 입술 길이 등을 계산
    def calculate_self(self, detection_time, printkey = (0,5), option = 0):
        """ printkey: ( 첫번째 index , 길이 ) """
        # 함수가 호출된 시간 저장
        self.time2 = detection_time

        # 계산함수 호출
        self.innermouth_aspect_ratio() # self.imar
        self.outtermouth_distfactor() # self.A,B,C..

        # update_specific_value
        if option == 0:
            self.specific_values[self.loop,:6] = [self.time2, self.imar,
                self.OB, self.OC, self.OD, self.OF]
        elif option == 1:
            if self.buf_top < self.buffersize:
                self.specific_values[self.buf_top,:6] = [self.time2, self.imar,
                    self.OB, self.OC, self.OD, self.OF]
                self.buf_top += 1
        # 터미널 출력
        p = lambda f, s: print(self.name, "의 값:\n", self.specific_values[f:f+s,:6])
        # p(printkey[0], printkey[1])

    # 몇가지 변수를 초기화
    def refresh(self, option):
        """ refresh(option)
        option: 'all', 'color', 'TOTAL_SUB' """
        if option == "all":
            self.TOTAL_SUB = 0
            self.color_a = 255
            self.color_b = 255
            self.probability = 0
            # self.Mouth_movement = 0
        elif option == "color":
            self.color_a = 255
            self.color_b = 255
        elif option == "TOTAL_SUB":
            self.TOTAL_SUB = 0

    # 화자 표시
    def masking(self, words):
        self.color_a = 50
        self.color_b = 50
        self.sayingwords = words
        return

    def show_box(self, frame):
        text = "{}".format(self.name)
        # Mouth_movement 출력 텍스트
        t2 = "{:.4f}".format(self.Mouth_movement)
        # t3 = self.man
        # 얼굴 테두리 사각형
        cv2.rectangle(frame, (self.sx, self.sy), (self.ex, self.ey), (0, 0, 255), 2)
        # 입술 주위 점
        for (x, y) in self.Inmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in self.Outmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        # 말풍선
        x, y = self.Midmark[0,0], self.Midmark[0,1]
        cv2.rectangle(frame, (x - 160, y - 150), (x + 160, y - 70), (255, 255, 255), -1)
        cv2.rectangle(frame, (x - 160, y - 150), (x + 160, y - 70), (self.color_a, 255, self.color_b), 3)
        cv2.putText(frame, text, (x-160, y-130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, t2, (x-130, y-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # cv2.putText(frame, t3, (x-200, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
