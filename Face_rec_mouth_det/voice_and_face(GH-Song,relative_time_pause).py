# import the necessary packages
from imutils.video import FileVideoStream, VideoStream, FPS
import imutils
import time
import cv2
import dlib
import os
from time import time as pytime
from datetime import datetime
from speakutils import speak_utils
from voiceutils import voice_utils
from nameutils import face_name_utils

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

recorded_words = "" # 녹음된 문장

# 분류 가능한 이름들
names = ["Song_GH", "Kim_JW", "Choi_EH"]
names_detected = []

# 프로그램 동작 여부
program_on = False
#####################################################################

# <editor-fold>
fps = FPS().start()
vcu = voice_utils("korean", "output/short_record.wav")
fnu = face_name_utils()
# </editor-fold>


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

    elif key == "o":
        program_on = False
        print("[INFO] 현재 TH_of_Movement: ", TH_of_Movement)
        TH_of_Movement = float(input("TH_of_Movement 조정: "))
    while program_on == True:
        # grab the frame from the threaded video stream
        # resize the frame to have a width of 600 pixels
        frame = imutils.resize(vs.read(), width=600)
        # make gray frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 음성을 버퍼에 저장
        vcu.mic_read()

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
                vcu.mic_read()

        # 직전의 검출 결과의 업데이트를 반영해서
        #print(TOTAL_SUB.items())
        # 만들어줘야 할 변수: starx-endy,
        if detections.shape[2] > 0:
            TOTAL_SUB_list = []
            for name in names_detected:
                TOTAL_SUB_list.append(man[name].TOTAL_SUB)
            #print("prob_list", sum(probability_list))
            #print( getTime(pytime(), First_time) )
            #print("rec_pause:", rec_pause)
            #print( getTime(pytime(), First_time) > rec_pause )
            if getTime(pytime(), First_time) > rec_pause:
                if sum(TOTAL_SUB_list) >= 1:
                # 누가 가장 입을 많이 움직였는가 확인할 것
                    for name in names_detected:
                        vcu.mic_read()
                        if man[name].TOTAL_SUB >= max(TOTAL_SUB_list):
                            # 화자 검출 성공
                            print("speaker detected")
                            [man[key].refresh("all") for key in names] # 중요 변수 초기화
                            # 음성인식
                            vcu.make_wavfile("clear")
                            recorded_words = vcu.request_STT()
                            # 화자임을 표시
                            man[name].masking(recorded_words)
                            print("[화자 검출 시점]", datetime.now(),':', name, "의 말:", man[name].current_sentence)
                            First_time = getTime(pytime())
                            rec_pause = 2
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

# stop the timer and display FPS information
fps.stop()
