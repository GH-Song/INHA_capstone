import extract_embeddings
import train_data
import face_cam

def FaceID():
    face_cam.face_from_cam()

    # 특징값 추출
    extract_embeddings.extract()

    # 학습
    train_data.image_training()

if __name__ == "__main__":
    # execute only if run as a script
    # 20장의 사진 저장
    FaceID()
