# import the necessary packages
from imutils.video import FileVideoStream, VideoStream, FPS
import imutils
import time
import cv2
import dlib
import numpy as np
import threading
import os
from time import time as pytime
from datetime import datetime
from speakutils import speak_utils
from voiceutils import voice_utils
from nameutils import face_name_utils
from FaceID_resistration import FaceID
from PIL import ImageFont, Image, ImageDraw

########################실행시 고려할 부분########################
# 기준값
TH_of_confidence = 0.6
TH_of_Movement = 0.15

# 녹음된 문장
recorded_words = ""

# 대화 기록
Total_words = ""

# 폰트
unicode_font = ImageFont.truetype('./gulim.ttf')

# 분류 가능한 이름들
namepath = os.path.join(os.getcwd(),"dataset")
names = os.listdir(namepath)
names.remove("Unknown")
names_detected = []

# 프로그램 동작 여부
program_on = False

toggle = True
finish = 1 # 녹음 종료 인지
#####################################################################

# <editor-fold>

def getTime(s, referencetime = 0):
    ss = s / 1 - referencetime
    return ss

def user_interface():
    key = input("[Options]\n"+
    "press 's' for start\n" +
    "press 'o' to change options\n"+
    "press 'r' to register new training data\n"
    "press 'q' for quit: \n>> ")
    return key

#vcu1 = voice_utils("korean", "output/short_record.wav")

#time.sleep(2)

# </editor-fold>
def Threading():
    while True:
        vcu.mic_read()


# loop over frames from the video file stream
while True:
    # wait for key in terminal
    key = user_interface()

    # 프로그램을 종료합니다
    if key == "q":
        print("program finished")
        break

    # 얼굴인식 성능을 높이기 위해 새로운 사진을 등록합니다
    elif key == "r":
        FaceID()

    # 대화 시각화 프로그램을 시작합니다.
    elif key == "s":
        program_on = True
        # initialize the video stream, then allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

        # fps측정 시작
        fps = FPS().start()

        # 음성인식 객체 생성
        vcu = voice_utils("korean", "output/short_record.wav")
        #vcu = voice_utils("english", "output/short_record.wav")

        # 얼굴 인식 객체 생성
        fnu = face_name_utils()

        # 사람 객체 생성
        man = {name: speak_utils(name, TH_of_Movement) for name in names}

        # 시간 동기화
        reftime = pytime()
        First_time = getTime(pytime(), reftime)
        Standard_time = First_time

        # 마이크 녹음 초기화
        vcu.mic_setup()

        t1 = threading.Thread(target=Threading)
        t1.start()



    # 역치값을 조정합니다
    elif key == "o":
        program_on = False
        print("[INFO] 현재 TH_of_Movement: ", TH_of_Movement)
        TH_of_Movement = float(input("TH_of_Movement 조정: "))

    while program_on == True:
        # grab the frame from the threaded video stream
        # resize the frame to have a width of 600 pixels
        frame = imutils.resize(vs.read(), width=800)
        # make gray frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(vcu.data2)

        # 음성을 버퍼에 저장
        #vcu.mic_read()

        # 이름 예측 클래스에 프레임 정보를 전달
        fnu.read_frame(frame)

        # 딥러닝 기반으로, 프레임에서 얼굴을 추출
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
                #vcu.mic_read()

        # 기준 시간 측정
        Last_time = getTime(pytime(), reftime)

        # 3초 이상 말하는 사람이 없을 시 wav 파일 초기화
        if (Last_time - Standard_time > 1):
            vcu.make_wavfile("clear")
            # print("[INFO] 대화가 인식되지 않습니다...")
            '''if toggle < 20:
                # 오랫동안
                # 음성을 버퍼에 저장
                vcu.mic_read()
                toggle += 1
            else:
                # 아무도 말을 안하는 것으로 간주
                print("[INFO] 대화가 인식되지 않습니다...")
                # 녹음파일 생성 - 스트림 버퍼 초기화
                vcu.make_wavfile("clear")
                toggle = 0'''

        # 모든 사람 인스턴스가 가진 self.TOTAL_SUB값을 리스트로 추출
        TOTAL_SUB_list = []
        for key in names_detected:
            TOTAL_SUB_list.append(man[key].TOTAL_SUB)

        if (Last_time - First_time > 0.5):

            # 누군가 한번 이상 말을 했으면 화자 검출
            if sum(TOTAL_SUB_list) >= 1:
                # 음성을 버퍼에 저장
                #vcu.mic_read()
                # 누가 가장 입을 많이 움직였는가 확인할 것
                for name in names_detected:
                    if man[name].TOTAL_SUB >= max(TOTAL_SUB_list):
                        # 화자 검출 성공
                        [man[key].refresh("all") for key in names] # 변수 초기화
                        First_time = getTime(pytime(), reftime) # 기준 시간 측정

                        # 말하는 시간 업뎃
                        Standard_time = First_time
                        finish = 0

                        # 화자임을 표시
                        current_speaker = name
                        man[name].masking("")
                        #print("[화자 검출 시점]", datetime.now(),':', name, "의 말:", man[name].current_sentence)
                        # toggle = 0
                        # 처음으로 화자가 된 시점 기록
                    else:
                        man[name].refresh("color")

            # 아무도 말을 하지 않았으면
            else:
                [man[key].refresh("color") for key in names]
                # 말이 끝날 때 데이터를 받아서 인식하게 만들기
                if finish == 0:
                    finish = 1
                    vcu.make_wavfile("clear")
                    print("[INFO] Recorded file is saved(화자 검출)")

                    recorded_words = vcu.request_STT()  # 클라우드 전송
                    Total_words += "\n(" + str(datetime.now())+ ') ' + current_speaker + " : " + vcu.request_STT()  # 대화 기록

                    man[current_speaker].masking(recorded_words)
                    #recorded_words = ""

                    print("[화자 검출 시점]", datetime.now(), ':', current_speaker, "의 말:",
                          man[current_speaker].current_sentence)

        ######################출력값 지정##############################
        if detections.shape[2] > 0:
            # 이름 출력 텍스트
            for name in names_detected:
                frame = man[name].draw_frame(frame)
            names_detected = []
        ###############################################################
        # update the FPS counter
        fps.update()
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("\n\n[INFO] Program is stopped...\n")
            f = open("record.txt", "w") # 텍스트 파일 만들기
            f.write(Total_words)
            f.close()
            vs.stop()
            cv2.destroyAllWindows()
            vcu.mic_setoff()
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            break
