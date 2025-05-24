from fastapi import FastAPI
from pydantic import BaseModel
from utils import analyze_emotion

app = FastAPI()

class EmotionRequest(BaseModel):
    utterance: str

@app.post("/analyze_emotion")
def analyze_emotion_endpoint(req: EmotionRequest):
    return analyze_emotion(req.utterance)