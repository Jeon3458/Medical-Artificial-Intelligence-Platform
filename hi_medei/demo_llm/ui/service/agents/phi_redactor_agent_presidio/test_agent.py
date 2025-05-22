import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run

input_text = {
    "text": "Hyunseong Jeon was admitted on June 29, 2000. His phone is 010-1234-5678."
}


result = run(input_text)
print(result)
