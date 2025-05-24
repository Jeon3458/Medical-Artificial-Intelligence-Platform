import requests

data = {
    "utterance": "요즘 너무 무기력하고 죽고 싶은 생각도 들어요"
}

res = requests.post("http://localhost:8000/analyze_emotion", json=data)
print(res.json())