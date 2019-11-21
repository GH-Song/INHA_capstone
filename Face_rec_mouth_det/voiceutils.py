import urllib3
import json
import base64
import pyaudio
import wave
from datetime import datetime

# pyaudio를 사용하여 음성 데이터를 받고 wav로 파일을 저장
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# 1024개의 샘플을 한 덩어리로 보고 잡음인지 음성인지 판단
CHUNK = 1024
RECORD_SECONDS = 2

openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
#accessKey = "210b1585-4ab9-46e3-9f94-5f0d0ca5d293" # 김장원
accessKey = "990540e9-e54e-4924-8ab8-4cc065b3354e" # 송국호

class voice_utils:
    """ 음성인식에 필요한 기능 """
    # 생성자
    def __init__(self, language, filename):
        # 음성인식 언어 지정
        self.languageCode = language
        # 녹음 파일 저장 경로
        self.audiofile = filename
        return

    # 마이크 스트리밍 시작
    def mic_setup(self):
        # 마이크 녹음 초기화
        print("[INFO] Setting microphone...")
        self.audio = pyaudio.PyAudio()
        self.audiostream = self.audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, input_device_index=1, frames_per_buffer=CHUNK)
        self.audioframe = []

    # 마이크 스트리밍 종료
    def mic_setoff(self):
        self.audiostream.stop_stream()
        self.audiostream.close()
        self.audio.terminate()

    # 클라우드 STT요청
    def request_STT(self):
        with open(self.audiofile, "rb") as f:
            self.audioContents = base64.b64encode(f.read()).decode("utf8")

        # 음성인식 기본값
        requestJson = {
            "access_key": accessKey,
            "argument": {
                "language_code": self.languageCode,
                "audio": self.audioContents
            }
        }

        http = urllib3.PoolManager()
        print("[INFO] 음성인식 요청중...")
        self.response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )

    def get_STT(self):
        data = self.response.data
        data = data.decode("utf-8")
        words = ""
        print("[responseCode] " + str(self.response.status))
        # print("음성인식 원본", data)

        # 한국어 출력
        if self.languageCode == "korean":
            # 원본 형식: {"result":0,"return_object":{"recognized":"ASR_NOTOKEN"}}
            words = data.replace('ASR_NOTOKEN', "")
            words = words.split('"')[7].strip()
        # 영어 출력
        elif self.languageCode == "english":
            # 원본 형식: {"result":0,"return_object":{"recognized":"  \n"}}
            words = data.replace('\\n',"")
            words = words.split('"')[7].strip()
        # 음성인식 결과 반환
        return words

    # 녹음 파일 저장
    def make_wavfile(self, mode = "clear"):
        waveFile = wave.open(self.audiofile, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.audioframe))
        waveFile.close()
        if mode == "clear":
            self.audioframe = [] # 버퍼 비우기, 중요

    # 마이크에서 소리를 읽어들이고 버퍼에 저장
    def mic_read(self):
        data = self.audiostream.read(CHUNK)
        self.audioframe.append(data)
