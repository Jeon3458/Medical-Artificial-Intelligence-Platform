from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "nlpai-lab/korean-emotion-classification"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_map = {
    0: "분노",
    1: "혐오",
    2: "불안",
    3: "슬픔",
    4: "기쁨",
    5: "당황",
    6: "상처"
}

def analyze_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    emotion = label_map[pred]
    risk_level = "낮음"
    flag = False
    comment = ""

    if emotion in ["슬픔", "불안", "상처", "분노"]:
        risk_level = "높음"
        flag = True
        comment = f"'{emotion}' 감정이 강하게 감지됩니다. 정서적 위험 가능성이 있습니다."
    else:
        comment = f"'{emotion}' 감정이 감지되었습니다. 비교적 안정적인 상태입니다."

    return {
        "emotion": emotion,
        "risk_level": risk_level,
        "flag": flag,
        "comment": comment
    }