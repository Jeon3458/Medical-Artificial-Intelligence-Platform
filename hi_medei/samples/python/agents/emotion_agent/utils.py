from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# KoELECTRA 기반 감정분석 모델 로딩 (가중치는 placeholder)
MODEL_NAME = "monologg/koelectra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# 라벨 매핑
label_map = {0: "중립", 1: "기쁨", 2: "우울"}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted = torch.argmax(logits, dim=1).item()
    label = label_map[predicted]

    risk_level = "낮음"
    flag = False
    comment = ""

    if label == "우울":
        risk_level = "높음"
        flag = True
        comment = "환자가 심각한 우울 상태일 수 있습니다. 추가 상담을 권장합니다."
    elif label == "기쁨":
        risk_level = "낮음"
        comment = "긍정적인 상태입니다."
    else:
        comment = "특별한 정서 위험은 감지되지 않았습니다."

    return {
        "emotion": label,
        "risk_level": risk_level,
        "flag": flag,
        "comment": comment
    }