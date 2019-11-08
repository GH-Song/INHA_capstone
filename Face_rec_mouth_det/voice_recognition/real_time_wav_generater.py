import urllib3
import json
import base64
import pyaudio
import wave

from time import time as pytime

reftime = pytime()
def getTime(s, referencetime = 0):
    ss = s / 1 - referencetime
    return ss

''' 10초 녹음 파일은 클라우드 분석에 약 3초 소요 '''

# RECORD_SECONDS를 화자 인식 조건을 결합하여 화자 인식이 한 사람에게 되는 동안만 음성 파일을 만들 생각임
# 그리고 나서 변환된 음성 데이터를 말풍선에 띄우는 식으로 갈 것. 따라서 데이터 변환 되는 데 걸리는 시간 만큼 화자 끼리 말하는 데 텀 시간이 필요.
# 실시간으로 음성 데이터가 바뀌어야 하므로 file 이름이 겹쳐 쓰여도 상관없으나 나중에 데이터 저장이나 문서화 하게 될 시 저장 공간도 따로 필요 할 것으로 생각 됨.
# 구글 음성 인식은 실시간 데이터를 판단 할 때 기준 점을 만들기가 쉽지 않고 최소 데이터가 15초로 인식되어 무료 60분이 금방 소모 될 것이라 판단. 그리고 구글 코드 분석이 너무 어려움.
# ETRI는 일 1000건 무료라 거의 오픈 소스라 생각해도 무방

# pyaudio를 사용하여 음성 데이터를 받고 wav로 파일을 저장해주는 코드
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# 1024개의 샘플을 한 덩어리로 보고 잡음인지 음성인지 판단하는 것임
CHUNK = 1024
RECORD_SECONDS = 10

# 음성 파일 이름 지정. 나중에 실시간으로 바꾸거나 데이터를 따로 저장 할 것임.
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording

stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, input_device_index=1, frames_per_buffer=CHUNK)

print("recording...")

frames = []

i = 0
while True:
    data = stream.read(CHUNK)
    frames.append(data)
    i += 1
    print(i)
    if i > 200:
        i = 0
        break

# stop Recording

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

print("파일생성, 클라우드 전송시간 ", getTime(pytime(), reftime))

# 만들어진 wav 파일을 바로 음성 인식을 통해 텍스트로 변환해줌
openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
# accessKey = "210b1585-4ab9-46e3-9f94-5f0d0ca5d293"
accessKey = "990540e9-e54e-4924-8ab8-4cc065b3354e" # 송국호
audioFilePath = "file.wav"

languageCode = "korean"

file = open(audioFilePath, "rb")
audioContents = base64.b64encode(file.read()).decode("utf8")
file.close()

requestJson = {
    "access_key": accessKey,
    "argument": {
        "language_code": languageCode,
        "audio": audioContents
    }
}

http = urllib3.PoolManager()
response = http.request(
    "POST",
    openApiURL,
    headers={"Content-Type": "application/json; charset=UTF-8"},
    body=json.dumps(requestJson)
)

data = response.data
data = data.decode("utf-8")

print("[responseCode] " + str(response.status))
print("[responBody]")
print("종료시간 ", getTime(pytime(), reftime))
print(data)
