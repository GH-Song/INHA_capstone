import urllib3
import json
import base64

openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
# key는 http://aiopen.etri.re.kr/index.php 에서 사용 신청해서 써야함
accessKey = "YOUR_ACCESS_KEY"
# 오디오 파일 위치 : 오디오 파일은 확장자는 .wav 이어야 하고 오디오 코덱은 PCM, 채널은 mono chanenl, 16Khz의 샘플링 레이트를 가져야함 그리고 1M byte를 초과하면 안됨 (20~30초 가량).
audioFilePath = "AUDIO_FILE_PATH"
# 원하는 language ex) "korean"
languageCode = "LANGUAGE_CODE"

file = open(audioFilePath, "rb")
# 오디오 파일을 binary code로 만들어줌
audioContents = base64.b64encode(file.read()).decode("utf8")
file.close()

# 요청 parameter
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

# 받은 utf-8 코드를 다시 한국어로 변환 해주는 작업
data = response.data
data = data.decode("utf-8")

print("[responseCode] " + str(response.status))
print("[responBody]")
print(reponse.data)
print(data)