import urllib3
import json
import base64
import pyaudio
import wave
from datetime import datetime
import re

# pyaudio를 사용하여 음성 데이터를 받고 wav로 파일을 저장
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# 1024개의 샘플을 한 덩어리로 보고 잡음인지 음성인지 판단
CHUNK = 1024
RECORD_SECONDS = 2

openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
#accessKey = "210b1585-4ab9-46e3-9f94-5f0d0ca5d293"
accessKey = "990540e9-e54e-4924-8ab8-4cc065b3354e"
# audioFilePath = "C:/Users/GHsong/Projectfiles/INHA_capstone/Face_rec_mouth_det/file.wav"
languageCode = "korean"
#languageCode = "english"
# 음성 파일 이름 지정. 나중에 실시간으로 바꾸거나 데이터를 따로 저장 할 것임.
WAVE_OUTPUT_FILENAME = "output/short_record.wav"

class voice_utils:
    """ 음성인식에 필요한 기능 """
    def __init__(self):
        return
    def mic_setup(self):
        # 마이크 녹음 초기화
        print("[INFO] Setting microphone...")
        self.audio = pyaudio.PyAudio()
        self.audiostream = self.audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, input_device_index=1, frames_per_buffer=CHUNK)
        self.audioframe = []

    def mic_setoff(self):
        self.audiostream.stop_stream()
        self.audiostream.close()
        self.audio.terminate()

    def request_STT(self):
        with open(WAVE_OUTPUT_FILENAME, "rb") as f:
            self.audioContents = base64.b64encode(f.read()).decode("utf8")

        # 음성인식 기본값
        requestJson = {
            "access_key": accessKey,
            "argument": {
                "language_code": languageCode,
                "audio": self.audioContents
            }
        }

        http = urllib3.PoolManager()
        print("[INFO] 음성인식 요청중...")
        response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )

        data = response.data
        data = data.decode("utf-8")
        words = ""
        print("[responseCode] " + str(response.status))
        print(data)
        if languageCode == "korean":
            #words = data.replace('\\n',"")
            words = data.replace('ASR_NOTOKEN', "")
            words = words.split('"')[7].strip()
            #  {"result":0,"return_object":{"recognized":"  \n"}}
        elif languageCode == "english":
            words = data.replace('\\n',"")
            words = words.split('"')[7].strip()
        return words

    def make_wavfile(self):
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.audioframe))
        waveFile.close()
        self.audioframe = []
        # os.remove(WAVE_OUTPUT_FILENAME)

    def mic_read(self):
        data = self.audiostream.read(CHUNK)
        self.audioframe.append(data)
