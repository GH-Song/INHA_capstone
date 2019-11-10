""" 얼굴인식 학습 과정을 자동으로 처리합니다 """

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

def image_training():
    path_embeddings = "output/embeddings.pickle" # path to serialized db of facial embeddings
    path_recognizer = "output/recognizer.pickle" # path to output model trained to recognize faces
    path_le = "output/le.pickle" # path to output label encoder

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(path_embeddings, "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(path_recognizer, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(path_le, "wb")
    f.write(pickle.dumps(le))
    f.close()

if __name__ == "__main__":
    # execute only if run as a script
    image_training()
