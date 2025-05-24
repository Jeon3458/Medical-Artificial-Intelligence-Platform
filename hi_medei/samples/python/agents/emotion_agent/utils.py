from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_map = {
    0: "우울",   # originally '부정'
    1: "중립",
    2: "긍정"
}

# 위험 키워드 목록 (무조건 emotion = "위험")
force_risk_keywords = ["죽고", "자살", "극단", "포기하고", "도망치고", "살기 싫어", "괴로워", "목숨", "생을", "끝내고","힘들"]

# 긍정 재정정 키워드 (중립/우울이면 긍정으로 바꿈)
positive_keywords = ["행복", "기쁘다", "즐겁다", "감사", "웃음", "설레", "좋은 하루", "친구들과", "신난다","뿌듯","재미","기뻐","기대대"]

def analyze_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    emotion = label_map.get(pred, "알 수 없음")
    prob = round(probs[0][pred].item(), 4)

    # 기본값
    risk_level = "낮음"
    flag = False
    comment = f"'{emotion}' 감정이 감지되었습니다. 확률: {prob}"

    # 위험 키워드 강제 판정
    if any(keyword in text for keyword in force_risk_keywords):
        emotion = "위험"
        risk_level = "최고"
        flag = True
        comment = f" 위험 키워드 감지됨 → 감정 재정정: '{emotion}', 위험도: {risk_level}"

    # 긍정 키워드 감지 시 덮어쓰기
    elif emotion in ["우울", "중립"] and any(kw in text for kw in positive_keywords):
        emotion = "긍정"
        risk_level = "낮음"
        flag = False
        comment = "문맥상 긍정 문장으로 인식되어 감정 재조정됨. 확률 무시됨."

    # 우울 기본 처리
    elif emotion == "우울":
        risk_level = "높음"
        flag = True
        comment += " → 정서적 위험 가능성이 있습니다."

    return {
        "emotion": emotion,
        "risk_level": risk_level,
        "flag": flag,
        "comment": comment
    }
