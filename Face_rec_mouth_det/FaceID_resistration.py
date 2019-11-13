import extract_embeddings
import train_data
import face_cam

def FaceID():
    face_cam.face_from_cam()

    userkey = input("촬영한 사진을 등록하시겠습니까? [y/n] >> ")

    if userkey == "y":
        # 특징값 추출
        extract_embeddings.extract()

        # 학습
        train_data.image_training()

if __name__ == "__main__":
    # execute only if run as a script
    # 20장의 사진 저장
    FaceID()
