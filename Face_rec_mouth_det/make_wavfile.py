import urllib3
import json
import base64
import pyaudio
import wave

# pyaudio를 사용하여 음성 데이터를 받고 wav로 파일을 저장해주는 코드
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# 1024개의 샘플을 한 덩어리로 보고 잡음인지 음성인지 판단하는 것임
CHUNK = 1024

RECORD_SECONDS = 2

# 음성 파일 이름 지정. 나중에 실시간으로 바꾸거나 데이터를 따로 저장 할 것임.
WAVE_OUTPUT_FILENAME = "file.wav"

def audio_recording():
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, input_device_index=1, frames_per_buffer=CHUNK)

    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("finished recording")

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
