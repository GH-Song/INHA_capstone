# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import imutils
import time
import cv2
import dlib
import os
import speakutils
from speakutils import speak_utils
from time import time as pytime
from datetime import datetime
import voiceutils
from voiceutils import voice_utils
import nameutils
from nameutils import face_name_utils

reftime = pytime()
def getTime(s, referencetime = 0):
    ss = s / 1 - referencetime
    return ss
########################실행시 고려할 부분########################
# 녹음 종료 인지
finish = 0
# 녹음된 문장
recorded_words = ""

# 기준값
TH_of_confidence = 0.6
TH_of_Movement = 0.2

# 분류 가능한 이름들
names = ["Song_GH", "Kim_JW", "Choi_EH"]
names_detected = []

# 프로그램 동작 여부
program_on = False
#####################################################################

# <editor-fold>

# grab the indexes of the facial landmarks for the left and
outmark_start, outmark_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inmark_start, inmark_end = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the FPS throughput estimator
fps = FPS().start()
# </editor-fold>

vcu = voice_utils()
fnu = face_name_utils()

# loop over frames from the video file stream
while True:
    # wait for key in terminal
    key = input("press 's' for start, 'o' to change options, 'q' for quit: \n")
    # if the `q` key was pressed, break from the loop

    if key == "q":
        print("program finished")
        break

    elif key == "s":
        program_on = True
        # initialize the video stream, then allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

        # 마이크 녹음 초기화
        vcu.mic_setup()

        # 사람 객체 생성
        man = {name: speak_utils(name, TH_of_Movement) for name in names}

        # 시간 동기화
        reftime = pytime()
        First_time = getTime(pytime(), reftime)
        first_recording_time = First_time

    elif key == "o":
        program_on = False
        print("[INFO] 현재 TH_of_Movement: ", TH_of_Movement)
        TH_of_Movement = float(input("TH_of_Movement 조정: "))

    while program_on == True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        # resize the frame to have a width of 600 pixels
        frame = imutils.resize(frame, width=600)
        # make gray frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 이름 예측 클래스에 프레임 정보를 전달
        fnu.read_frame(frame)

        # 음성을 버퍼에 저장
        vcu.mic_read()

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detections = fnu.detect_face()
        # 검출된 얼굴들에 대한 반복
        for i in range(0, detections.shape[2]):

            # 이름을 예측하여 얼굴 사각형 좌표와 함께 반환
            face_box_index = [0,0,0,0]
            name, face_box_index = fnu.predict_name(i, frame, TH_of_confidence)

            # 얼굴이 검출된 이름만 따로 저장 -> 이후 작업은 이 이름에 대해서만
            if name != "Unknown" and name in names:
                # 검출된 이름 목록에 추가
                names_detected.append(name)

                # self.inmarks, self.outmarks, self.midmark 값이 부여됨
                man[name].landmark(gray, *face_box_index)

                # 미분 전에 미리 해줘야 할 것들
                detection_time = getTime(pytime(), reftime)

                # specific values, time2 update
                man[name].calculate_self(detection_time)

                # 미분 함수 호출, self.Mouth_movement에 반영, 계산시 time1 초기화
                man[name].differential(0.25, 2) # time2 - time1이 0.01보다 큰지 고려해서 실행됨=

                # 음성을 버퍼에 저장
                vcu.mic_read()
        #### detection loop나오기 (while문과 동일 위치)
        # 직전의 검출 결과의 업데이트를 반영해서
        # 이름 기준으로 반복
        Last_time = getTime(pytime(), reftime)
        # 2초동안 들린 소리가 누구 것에 가까운지 판단
        if (Last_time - first_recording_time) > 2:
            # 2초 마다 현재까지 녹음된 것을 파일로 저장
            print("finished recording")
            vcu.make_wavfile()
            first_recording_time = getTime(pytime(), reftime)

        if (Last_time - First_time) > 2:
            # 이전 검사 완료 이후 x초가 지났으면
            TOTAL_SUB_list = []
            # 모든 사람 인스턴스가 가진 self.TOTAL_SUB값을 리스트로 추출
            for key in names_detected:
                TOTAL_SUB_list.append(man[key].TOTAL_SUB)

            # 화자 검출
            if sum(TOTAL_SUB_list) >= 1:
                # 누군가 한번 이상 말을 했으면
                finish = 0
                for name in names_detected:
                # 누가 가장 입을 많이 움직였는가 확인할 것
                    if man[name].TOTAL_SUB >= max(TOTAL_SUB_list):
                        # 화자 검출 성공
                        for key in names:
                            man[key].refresh("all") # 중요 변수 초기화
                        # 화자임을 표시
                        recorded_words = vcu.request_STT()
                        print(datetime.now(), ': ', recorded_words)

                        man[name].masking(recorded_words)

                        First_time = getTime(pytime(), reftime)
                        first_recording_time = First_time
                    else:
                        man[name].refresh("color")
            else:
                # 아무도 말을 하지 않았으면
                for name in names:
                    man[name].refresh("color")

                # 계속해서 텍스트를 새로 받는 것보다 말이 끝날을 때만 데이터를 받아서 인식하게 만듬
                if finish == 0 :
                    finish = 1
                    print(name, "의 말:", man[name].current_sentence)
        ######################출력값 지정##############################
        #print(TOTAL_SUB.items())
        # 만들어줘야 할 변수: starx-endy,
        if detections.shape[2] > 0:
            for name in names_detected:
                # 이름 출력 텍스트
                man[name].show_box(frame)
            names_detected = []
        ###############################################################
        # update the FPS counter
        fps.update()
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            vs.stop()
            cv2.destroyAllWindows()
            vcu.mic_setoff()
            break
fps.stop()
