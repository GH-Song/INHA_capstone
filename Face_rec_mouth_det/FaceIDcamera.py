# import
import cv2
from imutils.video import FileVideoStream, VideoStream, FPS
import imutils
import os
import time

face_cascade = cv2.CascadeClassifier('libdata/haarcascade_frontalface_alt.xml')
path_imwrite = ""

ID = int(input('''이름을 선택하세요(번호 입력)
[1] Song_GH
[2] Kim_JW
[3] Choi_EH
>>'''))

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

print(path_imwrite)

while True:
    frame = imutils.resize(vs.read(), width=1000)
    cv2.putText(frame, text, (x-160, y-130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        vs.stop()
        cv2.destroyAllWindows()
        break

    elif key == ord("s")
        n = 0
        # 얼굴을 움직이면서 총 사진이 30장 될 때까지 반복?
        while n < 30:
            time.sleep(0.1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
                cropped = frame[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
                cv2.imwrite(os.path.join(path_imwrite, str(n)+".png"), cropped)
                n += 1

    #frame = cv2.resize(frame, (300, 300))

#face_image =
