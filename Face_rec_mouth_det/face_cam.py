# import
import cv2
from imutils.video import FileVideoStream, VideoStream, FPS
import imutils
import os
import time

def face_from_cam():
    face_cascade = cv2.CascadeClassifier('libdata/haarcascade_frontalface_alt.xml')
    path_imwrite = ""

    while True:
        ID = int(input(
        "이름을 선택하세요(번호 입력), 사진 등록 종료는 -1\n"+
        "[1] Song_GH\n"+
        "[2] Kim_JW\n"+
        "[3] Choi_EH\n"+
        ">>"))
        if ID == -1:
            break
        if ID == 1:
            path_imwrite = "dataset/Song_GH"
        elif ID == 2:
            path_imwrite = "dataset/Kim_JW"
        elif ID == 3:
            path_imwrite = "dataset/Choi_EH"
        else:
            path_imwrite = "dataset/Unknown"

        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

        while True:

            frame = imutils.resize(vs.read(), width=1000)
            cv2.putText(frame, "to save picture, press 's'", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                vs.stop()
                cv2.destroyAllWindows()
                break

            elif key == ord("s"):
                n = 0
                # 얼굴을 움직이면서 총 사진이 30장 될 때까지 반복?
                while n < 20:
                    frame = imutils.resize(vs.read(), width=1000)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3,5)

                    for (x,y,w,h) in faces:
                        cropped = frame[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
                        cv2.imwrite(os.path.join(path_imwrite, str(n)+".png"), cropped)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
                        n += 1

                    text = "saved: "+str(n)+"/20"
                    cv2.putText(frame, text, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        vs.stop()
                        cv2.destroyAllWindows()
                        break
                        
if __name__ == "__main__":
    # execute only if run as a script
    face_from_cam()
