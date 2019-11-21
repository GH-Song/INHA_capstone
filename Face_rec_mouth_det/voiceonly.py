from voiceutils import voice_utils
from time import time as pytime

def period():

    # 음성인식 객체 생성
    vcu = voice_utils("korean", "output/short_record.wav")
    #vcu = voice_utils("english", "output/short_record.wav")
    
    # 마이크 녹음 초기화
    vcu.mic_setup()
    past_time = 0

    while True:
        present_time = pytime()
        if present_time - past_time < 5:
            # 음성을 버퍼에 저장
            vcu.mic_read()
        else:
            past_time = present_time
            print("음성인식 종료")
            vcu.make_wavfile("clear")
            vcu.request_STT()  # 클라우드 요청
            recorded_words = vcu.get_STT()
            print(recorded_words)
            present_time = pytime()
            print("분석소요시간:", present_time - past_time)
            past_time = present_time

def active():
    # 음성인식 객체 생성
    vcu = voice_utils("korean", "output/short_record.wav")
    #vcu = voice_utils("english", "output/short_record.wav")
    # 마이크 녹음 초기화
    vcu.mic_setup()
    past_time = 0

    while True:
        present_time = pytime()
        if present_time - past_time < 5:
            # 음성을 버퍼에 저장
            vcu.mic_read()
        else:
            past_time = present_time
            print("음성인식 종료")
            vcu.make_wavfile("clear")
            vcu.request_STT()  # 클라우드 요청
            recorded_words = vcu.get_STT()
            print(recorded_words)
            present_time = pytime()
            print("분석소요시간:", present_time - past_time)
            past_time = present_time
