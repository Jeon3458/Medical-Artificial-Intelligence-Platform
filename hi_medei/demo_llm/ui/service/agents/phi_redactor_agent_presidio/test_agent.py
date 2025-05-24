# test_korean.py
from main import run

input_text_ko = {
    "text": "전현성님은 2000년 6월 29일에 입원했습니다. 연락처는 010-1234-5678입니다."
}

result = run(input_text_ko)
print(result)