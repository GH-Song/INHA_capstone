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

# face landmark 추출에 필요한 predictor모듈 기능 활성화
PATH_predictor = "shape_predictor_68_face_landmarks.dat"
print("[INFO] loading facial landmark predictor in speakutils...")
predictor = dlib.shape_predictor(PATH_predictor)

outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# my class
class speak_utils:
    '''대화와 관련된 클래스'''
    def __init__(self, name, buffersize = 20):
        self.name = name
        self.buffersize = buffersize
        self.TOTAL_SUB = 0
        self.color_a = 255
        self.color_b = 255
        self.specific_values = np.zeros((self.buffersize,7), dtype = np.float64)
        self.time1 = 0;    self.time2 = 0
        self.Mouth_movement = 0
        self.sx = 0;    self.sy = 0
        self.ex = 0;    self.ey = 0
        self.Inmarks = []
        self.Outmarks = []
        self.Midmark = []
        self.timeset = True
        self.loop = 0
        self.probability = 0
        self.face_area = 0

    def landmark(self, gray, sx=0, sy=0, ex=0, ey=0):
        '''검출된 얼굴 좌표들을 저장'''
        # grab the indexes of the facial landmarks for the left and
        self.sx = int(sx);    self.sy = int(sy)
        self.ex = int(ex);    self.ey = int(ey)
        X = self.sx - self.ex
        Y = self.sy - self.ey
        self.face_area = abs(X*Y)
        rect = dlib.rectangle(left=self.sx, top=self.sy, right=self.ex, bottom=self.ey)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        self.Inmarks = shape[inmark_start:inmark_end]
        self.Outmarks = shape[outmark_start: outmark_end]
        self.Midmark = shape[28:29]
        return

    def speakerfind():
        # 이런 함수는 스태틱으로
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
        return
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

    def print_specific_values(self, data_start = 0, data_size = 5):
        data_end = data_start + data_size
        print(self.name, "의 값:\n", self.specific_values[data_start:data_end,:6])

    def update_specific_values(self, row=0):
        self.specific_values[row,:6] = [self.time2, self.imar, self.OB, self.OC, self.OD, self.OF]

    def difference_of_specific_values(self):
        # 행 1-0 2-1 3-2...
        difference = list(range(self.buffersize))
        for i in range(self.loop + 1):
            # 5회 반복시 loop -- 4 i는 0,1,2,3,4
            if i < self.loop:
                difference[i] = self.specific_values[i+1, :6] - self.specific_values[i, :6]
            else:
                difference[i] = self.specific_values[i, :6] - self.specific_values[i-1, :6]
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
                        (abs(del_values[i][2])+
                        abs(del_values[i][3])+
                        abs(del_values[i][4])+
                        abs(del_values[i][5])*5)/ self.face_area
                    )/ abs(del_values[i][0]*110)
                self.Mouth_movement = self.Mouth_movement / (self.loop+1)
                self.time1 = self.time2
                self.specific_values = np.zeros((self.buffersize,7), dtype = np.float64)
                return True # 이것이 반환되면 self.loop초기화
            else:
                return False
        else:
            return
    def calculate_self(self, detection_time, printkey = (0,5)):
        # self.face_area()
        self.time2 = detection_time
        self.innermouth_aspect_ratio() # self.imar
        self.outtermouth_distfactor() # self.A,B,C..
        self.update_specific_values(self.loop)
        if printkey == (0,5):
            self.print_specific_values()
        else:
            self.print_specific_values(printkey[0],printkey[1])
        return

    def refresh(self, option):
        if option == "all":
            self.TOTAL_SUB = 0
            self.color_a = 255
            self.color_b = 255
            self.probability = 0
            self.Mouth_movement = 0
        elif option == "color":
            self.color_a = 255
            self.color_b = 255
        elif option == "TOTAL_SUB" :
            self.TOTAL_SUB = 0
