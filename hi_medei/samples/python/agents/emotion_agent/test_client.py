import requests

data = {
    "utterance": "요즘 너무 힘들고 무기력해서 아무것도 하기 싫어요"
}

res = requests.post("http://localhost:8000/analyze_emotion", json=data)
print(res.json())