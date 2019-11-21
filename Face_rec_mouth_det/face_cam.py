# import
import cv2
from imutils.video import FileVideoStream, VideoStream, FPS
import imutils
import os
import time
import shutil

def face_from_cam():
    face_cascade = cv2.CascadeClassifier('libdata/haarcascade_frontalface_alt.xml')
    path_imwrite = ""

    while True:
        path_dataset = os.path.join(os.getcwd(),"dataset")
        names = os.listdir(path_dataset)
        names.remove("Unknown")
        paths_of_names = []
        num_users = len(names)

        # 등록된 사용자 목록 출력
        print("등록된 사용자 목록-----------------\n")
        for i, name in enumerate(names):
            paths_of_names.append(os.path.join(path_dataset, name))
            print("[{}] {}\n".format(i+1, name))
        print("[{}] {}\n".format(num_users+1, "신규 사용자 추가"))
        print("-----------------------------------\n")
        userkey = input(
            "사진을 등록할 사용자 번호를 입력하세요 "+
            "(사진 제거는 해당 번호의 음수 입력, 종료는 q)\n"+ ">> "
        )
        if userkey == "q":
            break

        try:
            ID = int(userkey)

            # 사용자 삭제
            if ID < 0:
                d = abs(ID)-1
                print("\n[warning]", "삭제 대상:", names[d])
                key = input("복구할 수 없습니다. 정말로 삭제하시겠습니까?[y/n] >> ")
                if key == "y":
                    shutil.rmtree(paths_of_names[d])
                    print("\n[INFO]", names[d], " 사용자를 삭제했습니다\n")
                continue
            else:
                ID = ID-1

            # 기존 사용자에 사진 추가
            if ID < num_users:
                path_imwrite = paths_of_names[ID]

            # 신규 등록
            elif ID == num_users:
                new = input("사용자 이름을 입력하세요>>")
                new_path = os.path.join(path_dataset, new)
                os.mkdir(new_path)
                path_imwrite = new_path

        except:
            print("\n[Error] 정상적인 입력이 아닙니다.\n")
            continue

        if path_imwrite == "":
            print("[warn] 사진을 저장할 대상이 없습니다.")

        else:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(1.0)

            while True:

                frame = imutils.resize(vs.read(), width=1000)

                # 안내문 출력
                cv2.putText(frame, "to save pictures, press 's'", (100, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
                cv2.putText(frame, "to save pictures, press 's'", (100, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
                cv2.putText(frame, "to quit camera, press 'q'", (100, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
                cv2.putText(frame, "to quit camera, press 'q'", (100, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    vs.stop()
                    cv2.destroyAllWindows()
                    break

                elif key == ord("s"):
                    n = 0
                    # 얼굴을 움직이면서 총 사진이 20장 될 때까지 반복
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
                        cv2.putText(frame, text, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
                        cv2.putText(frame, text, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q"):
                            break

                    cv2.destroyAllWindows()
                    print("[INFO] 촬영을 종료합니다. 저장된 사진: ", n,"장\n\n")
                    break
        vs.stop()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    # execute only if run as a script
    face_from_cam()
