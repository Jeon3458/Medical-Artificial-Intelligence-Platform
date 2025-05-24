from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import predict_emotion

app = FastAPI()

class EmotionRequest(BaseModel):
    utterance: str

@app.post("/analyze_emotion")
def analyze_emotion(request: EmotionRequest):
    result = predict_emotion(request.utterance)
    return result