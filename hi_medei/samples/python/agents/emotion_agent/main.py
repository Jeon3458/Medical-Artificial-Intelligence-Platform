from fastapi import FastAPI
from pydantic import BaseModel
from utils import analyze_emotion

app = FastAPI()

class A2ARequest(BaseModel):
    input: str

@app.post("/a2a")
def handle_a2a(req: A2ARequest):
    result = analyze_emotion(req.input)

    return {
        "status": "success",
        "output": {
            "emotion": result["emotion"],
            "risk_level": result["risk_level"],
            "flag": result["flag"],
            "comment": result["comment"],
            "recommendation": get_recommendation(result)
        }
    }

def get_recommendation(result: dict) -> str:
    if result["emotion"] == "위험":
        return "즉시 심리상담 연결 또는 전문기관 안내 필요"
    elif result["emotion"] == "우울":
        return "정서적 안정이 필요합니다. 상담을 권장합니다."
    elif result["emotion"] == "긍정":
        return "긍정적인 감정이 유지되도록 격려하세요."
    else:
        return "추가적인 정서 분석이 필요합니다."
