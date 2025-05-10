# medical_vector_db.py
import os
import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDataGenerator:
    """
    의료 데이터 생성기 - 벡터 DB 구축을 위한 풍부한 의료 데이터 생성
    """
    def __init__(self, output_dir="./medical_data"):
        """
        초기화 함수
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 환자 수 설정
        self.patient_counts = {
            "internal_medicine": 50,  # 내과 환자
            "surgery": 30,           # 외과 환자
            "cardiology": 25,        # 심장내과 환자
            "neurology": 20,         # 신경과 환자
            "emergency": 40,         # 응급 환자
        }
        
        # 나이 범위 설정
        self.age_ranges = {
            "internal_medicine": (35, 85),  # 내과는 고령 환자가 많음
            "surgery": (18, 75),            # 외과는 다양한 연령대
            "cardiology": (45, 90),         # 심장내과는 고령 환자 중심
            "neurology": (25, 80),          # 신경과는 중장년층부터 고령까지
            "emergency": (5, 95),           # 응급실은 전 연령대
        }
        
        # 의학 용어 사전 구축
        self._build_medical_dictionary()
        
        # 최근 날짜 범위 설정 (최근 1년)
        self.date_range_start = datetime.now() - timedelta(days=365)
        self.date_range_end = datetime.now()

    def _build_medical_dictionary(self):
        """
        의학 용어 사전 구축
        """
        # 진단명 (질병명)
        self.diagnosis_dict = {
            # 내과 질환
            "internal_medicine": [
                {"name": "고혈압", "icd10": "I10", "synonyms": ["hypertension", "혈압 상승", "본태성 고혈압"], 
                 "symptoms": ["두통", "어지러움", "시야 흐림"], 
                 "common_meds": ["암로디핀", "로자탄", "히드로클로로티아지드", "텔미사르탄"]},
                {"name": "제2형 당뇨병", "icd10": "E11", "synonyms": ["diabetes mellitus type 2", "성인형 당뇨병", "인슐린 비의존성 당뇨병"], 
                 "symptoms": ["다뇨", "다갈", "체중 감소", "피로", "시야 흐림"], 
                 "common_meds": ["메트포르민", "글리메피리드", "리나글립틴", "엠파글리플로진"]},
                {"name": "고지혈증", "icd10": "E78.5", "synonyms": ["hyperlipidemia", "고콜레스테롤혈증", "이상지질혈증"], 
                 "symptoms": ["무증상", "황색종"], 
                 "common_meds": ["아토르바스타틴", "로수바스타틴", "심바스타틴", "에제티미브"]},
                {"name": "위식도역류질환", "icd10": "K21", "synonyms": ["GERD", "gastroesophageal reflux disease", "역류성 식도염"], 
                 "symptoms": ["가슴쓰림", "산 역류", "흉통", "삼킴 곤란", "만성 기침"], 
                 "common_meds": ["에소메프라졸", "란소프라졸", "라베프라졸", "판토프라졸"]},
                {"name": "만성 폐쇄성 폐질환", "icd10": "J44", "synonyms": ["COPD", "chronic obstructive pulmonary disease", "만성 기관지염", "폐기종"], 
                 "symptoms": ["호흡곤란", "만성 기침", "가래 생성", "천명음", "흉부 압박감"], 
                 "common_meds": ["티오트로피움", "살메테롤/플루티카손", "포모테롤", "부데소니드"]},
                {"name": "간경변", "icd10": "K74", "synonyms": ["liver cirrhosis", "간경화", "간섬유화"], 
                 "symptoms": ["피로", "식욕감퇴", "복수", "황달", "다리 부종", "혼돈"], 
                 "common_meds": ["스피로놀락톤", "푸로세미드", "라크툴로스", "프로프라놀롤"]},
                {"name": "신부전", "icd10": "N18", "synonyms": ["kidney failure", "renal failure", "신장병"], 
                 "symptoms": ["소변량 감소", "부종", "피로", "호흡곤란", "메스꺼움"], 
                 "common_meds": ["푸로세미드", "칼시트리올", "세벨라머", "에리스로포이에틴"]},
                {"name": "류마티스 관절염", "icd10": "M05", "synonyms": ["rheumatoid arthritis", "RA", "만성 관절염"], 
                 "symptoms": ["관절통", "관절 부종", "관절 강직", "피로", "발열"], 
                 "common_meds": ["메토트렉세이트", "프레드니솔론", "레플루노미드", "아달리무맙"]},
            ],
            
            # 심장내과 질환
            "cardiology": [
                {"name": "관상동맥질환", "icd10": "I25", "synonyms": ["coronary artery disease", "CAD", "허혈성 심장질환", "심장동맥질환"], 
                 "symptoms": ["흉통", "호흡곤란", "심계항진", "발한", "어지러움"], 
                 "common_meds": ["아스피린", "클로피도그렐", "아토르바스타틴", "니트로글리세린"]},
                {"name": "심부전", "icd10": "I50", "synonyms": ["heart failure", "심장기능상실", "울혈성 심부전"], 
                 "symptoms": ["호흡곤란", "피로", "발목부종", "빠른 심박수", "기좌호흡"], 
                 "common_meds": ["푸로세미드", "에날라프릴", "카르베딜롤", "스피로놀락톤"]},
                {"name": "부정맥", "icd10": "I49", "synonyms": ["arrhythmia", "심장 리듬 장애", "심방 세동", "심실 빈맥"], 
                 "symptoms": ["심계항진", "가슴 두근거림", "호흡곤란", "어지러움", "실신"], 
                 "common_meds": ["아미오다론", "프로파페논", "소탈롤", "플레카이니드"]},
                {"name": "판막질환", "icd10": "I35", "synonyms": ["valvular heart disease", "대동맥판막증", "승모판막증"], 
                 "symptoms": ["호흡곤란", "피로", "흉통", "심계항진", "실신"], 
                 "common_meds": ["와파린", "베타차단제", "이뇨제"]},
                {"name": "심근경색", "icd10": "I21", "synonyms": ["myocardial infarction", "MI", "heart attack", "심장마비"], 
                 "symptoms": ["심한 흉통", "호흡곤란", "오심", "구토", "발한", "불안"], 
                 "common_meds": ["아스피린", "클로피도그렐", "에날라프릴", "메토프롤롤", "아토르바스타틴"]},
            ],
            
            # 신경과 질환
            "neurology": [
                {"name": "뇌졸중", "icd10": "I63", "synonyms": ["stroke", "cerebrovascular accident", "CVA", "뇌경색"], 
                 "symptoms": ["갑작스러운 마비", "언어장애", "시야장애", "심한 두통", "어지러움"], 
                 "common_meds": ["아스피린", "클로피도그렐", "와파린", "아토르바스타틴"]},
                {"name": "간질", "icd10": "G40", "synonyms": ["epilepsy", "seizure disorder", "경련성 질환"], 
                 "symptoms": ["발작", "의식 소실", "혼돈", "비정상적 행동", "감각 이상"], 
                 "common_meds": ["카바마제핀", "레베티라세탐", "라모트리진", "발프로산"]},
                {"name": "파킨슨병", "icd10": "G20", "synonyms": ["Parkinson's disease", "파킨슨증", "진전마비"], 
                 "symptoms": ["진전", "경직", "느린 움직임", "자세 불안정", "보행 장애"], 
                 "common_meds": ["레보도파", "프라미펙솔", "로피니롤", "라사길린"]},
                {"name": "알츠하이머병", "icd10": "G30", "synonyms": ["Alzheimer's disease", "치매", "노인성 치매"], 
                 "symptoms": ["기억력 감퇴", "판단력 저하", "언어 장애", "방향감각 상실", "성격 변화"], 
                 "common_meds": ["도네페질", "메만틴", "리바스티그민", "갈란타민"]},
                {"name": "다발성 경화증", "icd10": "G35", "synonyms": ["multiple sclerosis", "MS", "탈수초성 질환"], 
                 "symptoms": ["시력 문제", "감각 이상", "근력 약화", "균형 장애", "피로"], 
                 "common_meds": ["인터페론베타", "글라티라머 아세테이트", "핑골리모드", "나탈리주맙"]},
            ],
            
            # 외과 질환
            "surgery": [
                {"name": "급성 충수염", "icd10": "K35", "synonyms": ["acute appendicitis", "맹장염"], 
                 "symptoms": ["우하복부 통증", "오심", "구토", "발열", "식욕 감퇴"], 
                 "common_meds": ["항생제", "진통제"]},
                {"name": "담낭염", "icd10": "K81", "synonyms": ["cholecystitis", "쓸개염"], 
                 "symptoms": ["우상복부 통증", "발열", "오심", "구토", "황달"], 
                 "common_meds": ["항생제", "진통제", "제산제"]},
                {"name": "서혜부 탈장", "icd10": "K40", "synonyms": ["inguinal hernia", "사타구니 탈장"], 
                 "symptoms": ["사타구니 부위 돌출", "통증", "당김", "무거운 느낌"], 
                 "common_meds": ["진통제"]},
                {"name": "대장 게실염", "icd10": "K57", "synonyms": ["diverticulitis", "장 게실증"], 
                 "symptoms": ["좌하복부 통증", "발열", "오심", "변비 또는 설사", "복부 팽만"], 
                 "common_meds": ["항생제", "진통제", "변 연화제"]},
                {"name": "장폐색", "icd10": "K56", "synonyms": ["intestinal obstruction", "장 폐쇄", "장 막힘"], 
                 "symptoms": ["복통", "구토", "복부 팽만", "배변 곤란", "가스 배출 감소"], 
                 "common_meds": ["항생제", "진통제", "항구토제"]},
                {"name": "유방종양", "icd10": "D24", "synonyms": ["breast tumor", "유방 종괴", "유방암"], 
                 "symptoms": ["유방 덩어리", "유방 통증", "유두 분비물", "유방 피부 변화"], 
                 "common_meds": ["진통제", "항암제"]},
            ],
            
            # 응급 상황
            "emergency": [
                {"name": "심정지", "icd10": "I46", "synonyms": ["cardiac arrest", "심장 정지", "심폐 정지"], 
                 "symptoms": ["의식 소실", "호흡 정지", "맥박 없음"], 
                 "common_meds": ["에피네프린", "아미오다론", "리도카인"]},
                {"name": "호흡부전", "icd10": "J96", "synonyms": ["respiratory failure", "호흡 장애", "호흡 곤란"], 
                 "symptoms": ["호흡곤란", "청색증", "빠른 호흡", "의식 변화"], 
                 "common_meds": ["산소 요법", "기관지확장제", "스테로이드"]},
                {"name": "패혈증", "icd10": "A41", "synonyms": ["sepsis", "혈액 감염", "전신 염증 반응 증후군"], 
                 "symptoms": ["발열", "오한", "빠른 심박수", "빠른 호흡", "혼돈", "피부 발진"], 
                 "common_meds": ["광범위 항생제", "혈압 상승제", "수액 요법"]},
                {"name": "약물 과다복용", "icd10": "T50", "synonyms": ["drug overdose", "중독", "약물 중독"], 
                 "symptoms": ["의식 변화", "호흡 억제", "비정상 동공", "오심", "구토", "경련"], 
                 "common_meds": ["해독제", "활성탄", "지지 요법"]},
                {"name": "급성 뇌출혈", "icd10": "I61", "synonyms": ["acute cerebral hemorrhage", "뇌내 출혈", "뇌출혈"], 
                 "symptoms": ["극심한 두통", "의식 소실", "마비", "언어 장애", "구토"], 
                 "common_meds": ["만니톨", "항고혈압제", "항경련제"]},
                {"name": "다발성 외상", "icd10": "T07", "synonyms": ["multiple trauma", "다발성 손상", "복합 손상"], 
                 "symptoms": ["여러 부위 통증", "출혈", "호흡곤란", "의식 변화", "골절"], 
                 "common_meds": ["진통제", "항생제", "수액 요법", "혈압 상승제"]},
                {"name": "아나필락시스", "icd10": "T78.2", "synonyms": ["anaphylaxis", "과민성 쇼크", "알레르기 응급 반응"], 
                 "symptoms": ["호흡곤란", "두드러기", "안면 부종", "저혈압", "어지러움", "의식 소실"], 
                 "common_meds": ["에피네프린", "항히스타민제", "스테로이드", "산소 요법"]},
            ]
        }
        
        # 약물 정보
        self.medications = {
            # 심혈관계 약물
            "암로디핀": {"class": "칼슘 채널 차단제", "dosage": ["5mg", "10mg"], "frequency": ["1일 1회"], "purpose": "고혈압 치료"},
            "로자탄": {"class": "안지오텐신 II 수용체 차단제", "dosage": ["50mg", "100mg"], "frequency": ["1일 1회"], "purpose": "고혈압 치료"},
            "히드로클로로티아지드": {"class": "이뇨제", "dosage": ["12.5mg", "25mg"], "frequency": ["1일 1회"], "purpose": "고혈압 치료"},
            "아토르바스타틴": {"class": "스타틴", "dosage": ["10mg", "20mg", "40mg"], "frequency": ["1일 1회"], "purpose": "고지혈증 치료"},
            "아스피린": {"class": "항혈소판제", "dosage": ["100mg"], "frequency": ["1일 1회"], "purpose": "혈전 예방"},
            "클로피도그렐": {"class": "항혈소판제", "dosage": ["75mg"], "frequency": ["1일 1회"], "purpose": "혈전 예방"},
            "와파린": {"class": "항응고제", "dosage": ["2mg", "3mg", "5mg"], "frequency": ["1일 1회"], "purpose": "혈전 예방"},
            "푸로세미드": {"class": "이뇨제", "dosage": ["20mg", "40mg"], "frequency": ["1일 1-2회"], "purpose": "부종 치료, 심부전 치료"},
            "에날라프릴": {"class": "ACE 억제제", "dosage": ["5mg", "10mg", "20mg"], "frequency": ["1일 1-2회"], "purpose": "고혈압, 심부전 치료"},
            "메토프롤롤": {"class": "베타 차단제", "dosage": ["25mg", "50mg", "100mg"], "frequency": ["1일 1-2회"], "purpose": "고혈압, 심부전, 부정맥 치료"},
            
            # 내분비계 약물
            "메트포르민": {"class": "비구아니드", "dosage": ["500mg", "850mg", "1000mg"], "frequency": ["1일 2-3회"], "purpose": "당뇨병 치료"},
            "글리메피리드": {"class": "설포닐우레아", "dosage": ["1mg", "2mg", "4mg"], "frequency": ["1일 1회"], "purpose": "당뇨병 치료"},
            "리나글립틴": {"class": "DPP-4 억제제", "dosage": ["5mg"], "frequency": ["1일 1회"], "purpose": "당뇨병 치료"},
            "엠파글리플로진": {"class": "SGLT2 억제제", "dosage": ["10mg", "25mg"], "frequency": ["1일 1회"], "purpose": "당뇨병 치료"},
            "레보티록신": {"class": "갑상선 호르몬", "dosage": ["25mcg", "50mcg", "75mcg", "100mcg"], "frequency": ["1일 1회"], "purpose": "갑상선 기능 저하증 치료"},
            
            # 소화기계 약물
            "에소메프라졸": {"class": "양성자 펌프 억제제", "dosage": ["20mg", "40mg"], "frequency": ["1일 1회"], "purpose": "위식도역류질환 치료"},
            "라니티딘": {"class": "H2 수용체 길항제", "dosage": ["150mg", "300mg"], "frequency": ["1일 1-2회"], "purpose": "위산 분비 억제"},
            "오메프라졸": {"class": "양성자 펌프 억제제", "dosage": ["20mg", "40mg"], "frequency": ["1일 1회"], "purpose": "위궤양, 위식도역류질환 치료"},
            
            # 중추신경계 약물
            "레보도파": {"class": "도파민 전구체", "dosage": ["100mg", "250mg"], "frequency": ["1일 3-4회"], "purpose": "파킨슨병 치료"},
            "카바마제핀": {"class": "항경련제", "dosage": ["200mg", "400mg"], "frequency": ["1일 2-3회"], "purpose": "간질, 삼차신경통 치료"},
            "도네페질": {"class": "콜린에스테라제 억제제", "dosage": ["5mg", "10mg"], "frequency": ["1일 1회"], "purpose": "알츠하이머병 치료"},
            
            # 호흡기계 약물
            "살부타몰": {"class": "베타2 작용제", "dosage": ["2.5mg", "5mg"], "frequency": ["필요시"], "purpose": "천식, COPD 치료"},
            "플루티카손": {"class": "흡입 스테로이드", "dosage": ["50mcg", "100mcg", "250mcg"], "frequency": ["1일 2회"], "purpose": "천식, COPD 치료"},
            "몬테루카스트": {"class": "류코트리엔 수용체 길항제", "dosage": ["10mg"], "frequency": ["1일 1회"], "purpose": "천식 예방"},
            
            # 항생제
            "아목시실린": {"class": "페니실린계 항생제", "dosage": ["250mg", "500mg"], "frequency": ["1일 3회"], "purpose": "세균 감염 치료"},
            "세프트리악손": {"class": "세팔로스포린계 항생제", "dosage": ["1g", "2g"], "frequency": ["1일 1-2회"], "purpose": "중증 세균 감염 치료"},
            "레보플록사신": {"class": "퀴놀론계 항생제", "dosage": ["500mg", "750mg"], "frequency": ["1일 1회"], "purpose": "호흡기, 요로 감염 치료"},
            
            # 진통제 및 소염제
            "아세트아미노펜": {"class": "해열진통제", "dosage": ["325mg", "500mg", "650mg"], "frequency": ["1일 3-4회"], "purpose": "통증, 발열 완화"},
            "이부프로펜": {"class": "비스테로이드성 소염제", "dosage": ["200mg", "400mg", "600mg"], "frequency": ["1일 3-4회"], "purpose": "통증, 염증 완화"},
            "트라마돌": {"class": "마약성 진통제", "dosage": ["50mg", "100mg"], "frequency": ["1일 3-4회"], "purpose": "중등도~중증 통증 완화"},
            "프레드니솔론": {"class": "코르티코스테로이드", "dosage": ["5mg", "10mg", "20mg"], "frequency": ["1일 1회"], "purpose": "염증, 자가면역질환 치료"},
        }
        
        # 검사 항목
        self.lab_tests = {
            # 혈액 검사
            "혈색소": {"unit": "g/dL", "normal_range": {"남": "13-17", "여": "12-15"}, "abnormal_conditions": ["빈혈", "다혈구증"]},
            "백혈구": {"unit": "10^3/μL", "normal_range": {"남": "4.0-10.0", "여": "4.0-10.0"}, "abnormal_conditions": ["감염", "염증", "백혈병"]},
            "혈소판": {"unit": "10^3/μL", "normal_range": {"남": "150-450", "여": "150-450"}, "abnormal_conditions": ["혈소판감소증", "혈소판증가증"]},
            "AST": {"unit": "U/L", "normal_range": {"남": "0-40", "여": "0-32"}, "abnormal_conditions": ["간 손상", "근육 손상"]},
            "ALT": {"unit": "U/L", "normal_range": {"남": "0-41", "여": "0-33"}, "abnormal_conditions": ["간 손상", "약물 독성"]},
            "총 콜레스테롤": {"unit": "mg/dL", "normal_range": {"남": "125-200", "여": "125-200"}, "abnormal_conditions": ["고지혈증", "관상동맥질환 위험"]},
            "LDL 콜레스테롤": {"unit": "mg/dL", "normal_range": {"남": "0-130", "여": "0-130"}, "abnormal_conditions": ["고지혈증", "관상동맥질환 위험"]},
            "HDL 콜레스테롤": {"unit": "mg/dL", "normal_range": {"남": "40-60", "여": "50-60"}, "abnormal_conditions": ["관상동맥질환 위험"]},
            "중성지방": {"unit": "mg/dL", "normal_range": {"남": "0-150", "여": "0-150"}, "abnormal_conditions": ["고지혈증", "대사증후군"]},
            "공복혈당": {"unit": "mg/dL", "normal_range": {"남": "70-100", "여": "70-100"}, "abnormal_conditions": ["당뇨병", "저혈당증"]},
            "당화혈색소": {"unit": "%", "normal_range": {"남": "4.0-5.6", "여": "4.0-5.6"}, "abnormal_conditions": ["당뇨병", "혈당 조절 불량"]},
            "BUN": {"unit": "mg/dL", "normal_range": {"남": "8-20", "여": "8-20"}, "abnormal_conditions": ["신부전", "탈수"]},
            "크레아티닌": {"unit": "mg/dL", "normal_range": {"남": "0.6-1.2", "여": "0.5-1.1"}, "abnormal_conditions": ["신부전", "근육량 변화"]},
            "나트륨": {"unit": "mmol/L", "normal_range": {"남": "135-145", "여": "135-145"}, "abnormal_conditions": ["저나트륨혈증", "고나트륨혈증"]},
            "칼륨": {"unit": "mmol/L", "normal_range": {"남": "3.5-5.0", "여": "3.5-5.0"}, "abnormal_conditions": ["저칼륨혈증", "고칼륨혈증"]},
            "CRP": {"unit": "mg/L", "normal_range": {"남": "0-10", "여": "0-10"}, "abnormal_conditions": ["염증", "감염", "자가면역질환"]},
            "ESR": {"unit": "mm/hr", "normal_range": {"남": "0-15", "여": "0-20"}, "abnormal_conditions": ["염증", "감염", "자가면역질환"]},
            
            # 소변 검사
            "단백뇨": {"unit": "mg/dL", "normal_range": {"남": "음성", "여": "음성"}, "abnormal_conditions": ["신장 질환", "고혈압", "당뇨병"]},
            "적혈구뇨": {"unit": "/HPF", "normal_range": {"남": "0-4", "여": "0-4"}, "abnormal_conditions": ["요로 감염", "신장 결석", "신장 질환"]},
            "백혈구뇨": {"unit": "/HPF", "normal_range": {"남": "0-4", "여": "0-4"}, "abnormal_conditions": ["요로 감염", "방광염"]},
            "케톤체": {"unit": "", "normal_range": {"남": "음성", "여": "음성"}, "abnormal_conditions": ["당뇨병 케톤산증", "기아", "저탄수화물 식이"]},
            
            # medical_vector_db.py (계속)
            # 심장 검사
            "CK-MB": {"unit": "ng/mL", "normal_range": {"남": "0-5", "여": "0-5"}, "abnormal_conditions": ["심근경색", "근육 손상"]},
            "트로포닌 I": {"unit": "ng/mL", "normal_range": {"남": "<0.04", "여": "<0.04"}, "abnormal_conditions": ["심근경색"]},
            "BNP": {"unit": "pg/mL", "normal_range": {"남": "<100", "여": "<100"}, "abnormal_conditions": ["심부전", "폐색전증"]},
            "NT-proBNP": {"unit": "pg/mL", "normal_range": {"남": "<300", "여": "<300"}, "abnormal_conditions": ["심부전", "폐색전증"]},
            
            # 간 검사
            "총 빌리루빈": {"unit": "mg/dL", "normal_range": {"남": "0.1-1.2", "여": "0.1-1.2"}, "abnormal_conditions": ["간질환", "담도 폐쇄", "용혈성 빈혈"]},
            "직접 빌리루빈": {"unit": "mg/dL", "normal_range": {"남": "0-0.3", "여": "0-0.3"}, "abnormal_conditions": ["간질환", "담도 폐쇄"]},
            "ALP": {"unit": "U/L", "normal_range": {"남": "40-130", "여": "35-105"}, "abnormal_conditions": ["간담도 질환", "골질환"]},
            "GGT": {"unit": "U/L", "normal_range": {"남": "10-71", "여": "6-42"}, "abnormal_conditions": ["간담도 질환", "알코올 남용"]},
            
            # 갑상선 검사
            "TSH": {"unit": "μIU/mL", "normal_range": {"남": "0.4-4.0", "여": "0.4-4.0"}, "abnormal_conditions": ["갑상선기능저하증", "갑상선기능항진증"]},
            "Free T4": {"unit": "ng/dL", "normal_range": {"남": "0.7-1.9", "여": "0.7-1.9"}, "abnormal_conditions": ["갑상선기능저하증", "갑상선기능항진증"]},
            "T3": {"unit": "ng/dL", "normal_range": {"남": "80-200", "여": "80-200"}, "abnormal_conditions": ["갑상선기능항진증"]},
        }
        
        # 영상 검사
        self.imaging_tests = {
            "흉부 X선": {"common_findings": ["폐렴", "기흉", "심비대", "폐결절", "폐부종", "무기폐", "늑골 골절"]},
            "복부 X선": {"common_findings": ["장폐색", "복수", "장 가스 증가", "복강내 유리 가스", "이물질"]},
            "CT 흉부": {"common_findings": ["폐결절", "폐암", "폐색전증", "기관지 확장증", "간질성 폐질환", "흉막 삼출"]},
            "CT 복부": {"common_findings": ["간 병변", "신장 결석", "췌장염", "담석", "충수염", "대장 종양", "복강내 림프절 비대"]},
            "CT 두부": {"common_findings": ["뇌출혈", "뇌경색", "뇌종양", "외상성 뇌 손상", "뇌부종", "수두증"]},
            "MRI 뇌": {"common_findings": ["뇌졸중", "뇌종양", "다발성 경화증", "뇌염", "뇌동맥류", "뇌병변"]},
            "MRI 척추": {"common_findings": ["추간판 탈출증", "척추 협착증", "척추 종양", "척추 압박 골절", "척수 병변"]},
            "초음파 복부": {"common_findings": ["간 낭종", "간 지방증", "담석", "신장 결석", "담낭염", "췌장염"]},
            "초음파 심장": {"common_findings": ["판막 질환", "심장벽 운동 이상", "심실 비대", "심낭 삼출", "심방 확장"]},
            "혈관조영술": {"common_findings": ["관상동맥 협착", "혈관 폐색", "동맥류", "혈관 기형", "혈관 염증"]},
        }
        
        # 시술 및 수술
        self.procedures = {
            "관상동맥 중재술": {"description": "협착된 관상동맥을 풍선 확장 또는 스텐트로 넓히는 시술", "complications": ["재협착", "출혈", "혈관 손상"], "indications": ["관상동맥질환", "협심증", "심근경색"]},
            "담낭 절제술": {"description": "담낭을 제거하는 수술", "complications": ["감염", "출혈", "담도 손상"], "indications": ["담석증", "담낭염", "담낭 용종"]},
            "충수 절제술": {"description": "충수를 제거하는 수술", "complications": ["감염", "장폐색", "상처 문제"], "indications": ["급성 충수염"]},
            "대장 내시경": {"description": "직장부터 맹장까지 대장을 관찰하는 내시경 검사", "complications": ["천공", "출혈", "감염"], "indications": ["대장암 선별", "염증성 장질환", "변비", "설사", "직장 출혈"]},
            "위 내시경": {"description": "식도, 위, 십이지장을 관찰하는 내시경 검사", "complications": ["천공", "출혈", "인후통"], "indications": ["소화불량", "복통", "위식도역류", "위염", "위궤양"]},
            "척추 융합술": {"description": "척추를 고정시키는 수술", "complications": ["감염", "인접 분절 변성", "하드웨어 실패"], "indications": ["척추 불안정성", "추간판 질환", "척추 협착증"]},
            "유방 절제술": {"description": "유방 조직의 일부 또는 전체를 제거하는 수술", "complications": ["감염", "혈종", "림프부종"], "indications": ["유방암", "유방 종양"]},
            "인공 관절 치환술": {"description": "손상된 관절을 인공 관절로 대체하는 수술", "complications": ["감염", "인공관절 이완", "탈구"], "indications": ["관절염", "고관절 골절", "관절 변성"]},
            "뇌종양 절제술": {"description": "뇌종양을 제거하는 수술", "complications": ["뇌출혈", "뇌부종", "경련"], "indications": ["뇌종양", "뇌압 상승"]},
            "기관 삽관": {"description": "기도 확보를 위해 기관 내 튜브를 삽입하는 시술", "complications": ["저산소증", "기관 손상", "감염"], "indications": ["호흡 부전", "기도 폐쇄", "전신 마취"]},
        }
        
        # 의사 명단
        self.doctors = {
            "internal_medicine": [
                {"name": "김내과", "specialty": "내과", "id": "D001", "years_experience": 15},
                {"name": "이내과", "specialty": "내과", "id": "D002", "years_experience": 8},
                {"name": "박내과", "specialty": "내과", "id": "D003", "years_experience": 20},
            ],
            "cardiology": [
                {"name": "정심장", "specialty": "심장내과", "id": "D101", "years_experience": 12},
                {"name": "강심장", "specialty": "심장내과", "id": "D102", "years_experience": 7},
            ],
            "neurology": [
                {"name": "최신경", "specialty": "신경과", "id": "D201", "years_experience": 18},
                {"name": "윤신경", "specialty": "신경과", "id": "D202", "years_experience": 11},
            ],
            "surgery": [
                {"name": "서외과", "specialty": "외과", "id": "D301", "years_experience": 22},
                {"name": "한외과", "specialty": "외과", "id": "D302", "years_experience": 14},
                {"name": "임외과", "specialty": "외과", "id": "D303", "years_experience": 9},
            ],
            "emergency": [
                {"name": "조응급", "specialty": "응급의학과", "id": "D401", "years_experience": 10},
                {"name": "배응급", "specialty": "응급의학과", "id": "D402", "years_experience": 6},
            ],
        }

    def generate_patient(self, department, patient_id=None):
        """
        환자 기본 정보 생성
        """
        if patient_id is None:
            dept_code = department[0].upper()
            patient_id = f"{dept_code}{random.randint(10000, 99999)}"
        
        age_range = self.age_ranges.get(department, (20, 80))
        age = random.randint(age_range[0], age_range[1])
        
        gender = random.choice(["남", "여"])
        
        # 생년월일 계산 (현재 연도에서 나이를 빼서)
        current_year = datetime.now().year
        birth_year = current_year - age
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)  # 간단하게 28일로 제한
        
        # 주소 생성
        cities = ["서울", "부산", "인천", "대구", "대전", "광주", "울산", "세종", "수원", "안양", "성남", "고양", "용인", "천안", "청주", "전주", "포항"]
        districts = ["중구", "동구", "서구", "남구", "북구", "강남구", "강북구", "강동구", "강서구", "종로구", "중랑구", "노원구", "양천구", "마포구", "구로구"]
        address = f"{random.choice(cities)} {random.choice(districts)} "
        
        # 연락처
        phone = f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        
        # 보험 정보
        insurance_types = ["국민건강보험", "의료급여", "자동차보험", "산재보험", "사보험"]
        insurance = random.choice(insurance_types)
        
        # 알레르기 정보
        allergies = []
        if random.random() < 0.2:  # 20% 확률로 알레르기 있음
            possible_allergies = ["페니실린", "설파제", "아스피린", "요오드", "라텍스", "계란", "땅콩", "조개류", "밀가루", "우유"]
            num_allergies = random.randint(1, 3)
            allergies = random.sample(possible_allergies, num_allergies)
        
        # 흡연 상태
        smoking_status = random.choice(["비흡연", "과거 흡연", "현재 흡연"])
        if smoking_status == "과거 흡연":
            smoking_details = f"{random.randint(5, 30)}갑년, {random.randint(1, 10)}년 전 금연"
        elif smoking_status == "현재 흡연":
            smoking_details = f"{random.randint(5, 30)}갑년, 하루 {random.randint(5, 40)}개비"
        else:
            smoking_details = ""
        
        # 음주 상태
        alcohol_status = random.choice(["비음주", "사회적 음주", "과도한 음주"])
        if alcohol_status == "사회적 음주":
            alcohol_details = f"주 {random.randint(1, 2)}회, 소주 {random.randint(1, 3)}잔"
        elif alcohol_status == "과도한 음주":
            alcohol_details = f"주 {random.randint(3, 7)}회, 소주 {random.randint(4, 10)}잔"
        else:
            alcohol_details = ""
        
        # 체중 및 신장 (BMI 고려)
        if gender == "남":
            height = random.randint(160, 185)
            bmi = random.uniform(18.5, 30.0)
        else:
            height = random.randint(150, 175)
            bmi = random.uniform(18.0, 29.0)
        
        weight = round(bmi * (height/100) ** 2, 1)
        
        patient = {
            "id": patient_id,
            "name": f"{'김' if gender=='남' else '이'}{random.choice('가나다라마바사아자차카타파하')}{'돌' if gender=='남' else '미'}",
            "gender": gender,
            "birthdate": f"{birth_year}-{birth_month:02d}-{birth_day:02d}",
            "age": age,
            "address": address,
            "phone": phone,
            "insurance": insurance,
            "blood_type": random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
            "height": height,
            "weight": weight,
            "bmi": round(bmi, 1),
            "allergies": allergies,
            "smoking": {"status": smoking_status, "details": smoking_details},
            "alcohol": {"status": alcohol_status, "details": alcohol_details},
            "department": department,
            "emergency_contact": {
                "name": f"{'박' if gender=='남' else '최'}{random.choice('가나다라마바사아자차카타파하')}{'수' if gender=='남' else '영'}",
                "relationship": random.choice(["배우자", "자녀", "부모", "형제자매"]),
                "phone": f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
            }
        }
        
        return patient

    def generate_diagnoses(self, patient, count=None):
        """
        진단 정보 생성
        """
        if count is None:
            # 0-3개의 진단
            count = random.choices([0, 1, 2, 3], weights=[0.1, 0.5, 0.3, 0.1])[0]
        
        diagnoses = []
        department = patient["department"]
        
        if department not in self.diagnosis_dict:
            logger.warning(f"Unknown department: {department}, using internal_medicine instead")
            department = "internal_medicine"
        
        disease_list = self.diagnosis_dict[department]
        
        if not disease_list:
            return diagnoses
        
        selected_diseases = random.sample(disease_list, min(count, len(disease_list)))
        
        for disease in selected_diseases:
            # 진단일은 과거 날짜 (최대 3년 전까지)
            days_ago = random.randint(1, 1095)  # 최대 3년(1095일) 전
            diagnosis_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            doctor = random.choice(self.doctors.get(department, self.doctors["internal_medicine"]))
            
            # 진단 신뢰도 및 상태
            confidence = random.choice(["확정", "의증", "추정", "배제 필요"])
            if confidence == "확정":
                status = random.choice(["활성", "관해", "완치"])
            else:
                status = "평가 중"
            
            # 진단 메모
            severity_levels = ["경증", "중등도", "중증"]
            severity = random.choice(severity_levels)
            
            # 진단 메모 생성
            memo_options = [
                f"{severity} 수준의 {disease['name']}. 추가 검사 필요.",
                f"{disease['name']}으로 진단. {severity} 단계. 약물 치료 시작.",
                f"{disease['name']} {confidence}. {random.choice(disease['symptoms'])} 증상 있음.",
                f"{disease['name']} {status}. 정기적인 모니터링 필요.",
            ]
            memo = random.choice(memo_options)
            
            diagnoses.append({
                "name": disease["name"],
                "icd10": disease["icd10"],
                "date": diagnosis_date,
                "doctor": doctor["name"],
                "doctor_id": doctor["id"],
                "confidence": confidence,
                "status": status,
                "severity": severity,
                "memo": memo,
                "symptoms": random.sample(disease["symptoms"], min(len(disease["symptoms"]), random.randint(1, 3)))
            })
        
        return diagnoses

    def generate_medications(self, patient):
        """
        약물 처방 정보 생성
        """
        if "diagnoses" not in patient or not patient["diagnoses"]:
            return []
        
        medications = []
        
        for diagnosis in patient["diagnoses"]:
            disease = diagnosis["name"]
            
            # 진단별 적합한 약물 선택
            suitable_meds = []
            for disease_info in self.diagnosis_dict.get(patient["department"], []):
                if disease_info["name"] == disease and "common_meds" in disease_info:
                    suitable_meds = disease_info["common_meds"]
                    break
            
            if not suitable_meds:
                continue
            
            # 1-3개의 약물 선택
            selected_meds = random.sample(suitable_meds, min(random.randint(1, 3), len(suitable_meds)))
            
            for med_name in selected_meds:
                if med_name not in self.medications:
                    continue
                
                med_info = self.medications[med_name]
                
                # 처방일은 진단일과 같거나 이후
                diagnosis_date = datetime.strptime(diagnosis["date"], "%Y-%m-%d")
                days_after = random.randint(0, 30)
                prescription_date = (diagnosis_date + timedelta(days=days_after)).strftime("%Y-%m-%d")
                
                # 처방 기간
                duration = random.choice([7, 14, 28, 30, 60, 90])
                
                # 복용법
                dosage = random.choice(med_info["dosage"])
                frequency = random.choice(med_info["frequency"])
                
                medications.append({
                    "medication": med_name,
                    "class": med_info["class"],
                    "prescription_date": prescription_date,
                    "duration_days": duration,
                    "dosage": dosage,
                    "frequency": frequency,
                    "refill": random.randint(0, 3),
                    "purpose": med_info["purpose"],
                    "doctor": diagnosis["doctor"],
                    "doctor_id": diagnosis["doctor_id"],
                    "related_diagnosis": disease,
                    "special_instructions": random.choice([
                        "식후 복용", "식전 복용", "취침 전 복용", 
                        "필요시 복용", "증상 있을 때만 복용", ""
                    ])
                })
        
        return medications

    def generate_lab_results(self, patient):
        """
        검사 결과 생성
        """
        # 환자당 0-5회의 검사 결과
        num_tests = random.randint(0, 5)
        lab_results = []
        
        # 각 검사 기록마다
        for _ in range(num_tests):
            # 검사일은 과거 날짜
            days_ago = random.randint(1, 365)  # 최대 1년 전
            test_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # 검사 유형 선택 (일반 검사 vs. 특화된 검사)
            if "diagnoses" in patient and patient["diagnoses"]:
                # 진단이 있는 경우, 해당 진단과 관련된 검사를 더 높은 확률로 포함
                diagnosis_names = [d["name"] for d in patient["diagnoses"]]
                
                if any(name in ["고혈압", "심부전", "관상동맥질환"] for name in diagnosis_names):
                    test_types = ["기본 혈액검사", "심장 효소 검사", "지질 프로필"]
                elif any(name in ["당뇨병", "고지혈증"] for name in diagnosis_names):
                    test_types = ["기본 혈액검사", "당뇨 검사", "지질 프로필"]
                elif any(name in ["간경변", "간염"] for name in diagnosis_names):
                    test_types = ["기본 혈액검사", "간 기능 검사"]
                elif any(name in ["신부전", "신장병"] for name in diagnosis_names):
                    test_types = ["기본 혈액검사", "신장 기능 검사", "소변 검사"]
                else:
                    test_types = ["기본 혈액검사"]
            else:
                test_types = ["기본 혈액검사"]
            
            test_type = random.choice(test_types)
            
            # 검사 항목 선택
            if test_type == "기본 혈액검사":
                test_items = ["혈색소", "백혈구", "혈소판", "AST", "ALT", "BUN", "크레아티닌"]
            elif test_type == "심장 효소 검사":
                test_items = ["CK-MB", "트로포닌 I", "BNP", "NT-proBNP"]
            elif test_type == "지질 프로필":
                test_items = ["총 콜레스테롤", "LDL 콜레스테롤", "HDL 콜레스테롤", "중성지방"]
            elif test_type == "당뇨 검사":
                test_items = ["공복혈당", "당화혈색소", "인슐린"]
            elif test_type == "간 기능 검사":
                test_items = ["AST", "ALT", "총 빌리루빈", "직접 빌리루빈", "ALP", "GGT"]
            elif test_type == "신장 기능 검사":
                test_items = ["BUN", "크레아티닌", "나트륨", "칼륨"]
            elif test_type == "소변 검사":
                test_items = ["단백뇨", "적혈구뇨", "백혈구뇨", "케톤체"]
            elif test_type == "갑상선 검사":
                test_items = ["TSH", "Free T4", "T3"]
            else:
                test_items = ["혈색소", "백혈구", "혈소판"]
            
            # 검사 결과 값 생성
            results = {}
            abnormal_count = 0  # 비정상 결과 개수
            
            for item in test_items:
                if item not in self.lab_tests:
                    continue
                
                test_info = self.lab_tests[item]
                gender = patient["gender"]
                
                # 정상 범위 파싱
                normal_range = test_info["normal_range"].get(gender, "0-0")
                try:
                    if "-" in normal_range:
                        normal_min, normal_max = map(float, normal_range.split("-"))
                    elif "<" in normal_range:
                        normal_min, normal_max = 0, float(normal_range.replace("<", ""))
                    else:
                        normal_min, normal_max = 0, 0
                except:
                    normal_min, normal_max = 0, 0
                
                # 환자의 진단에 따라 비정상 결과 확률 조정
                abnormal_prob = 0.2  # 기본 20% 확률
                
                if "diagnoses" in patient:
                    for diagnosis in patient["diagnoses"]:
                        # 관련 질환에 따라 특정 검사 이상 확률 증가
                        if diagnosis["name"] == "당뇨병" and item in ["공복혈당", "당화혈색소"]:
                            abnormal_prob = 0.9
                        elif diagnosis["name"] == "고혈압" and item in ["BUN", "크레아티닌"]:
                            abnormal_prob = 0.6
                        elif diagnosis["name"] == "고지혈증" and item in ["총 콜레스테롤", "LDL 콜레스테롤", "중성지방"]:
                            abnormal_prob = 0.8
                        elif diagnosis["name"] == "간경변" and item in ["AST", "ALT", "총 빌리루빈"]:
                            abnormal_prob = 0.85
                        elif diagnosis["name"] == "신부전" and item in ["BUN", "크레아티닌"]:
                            abnormal_prob = 0.95
                
                # 값 생성
                is_abnormal = random.random() < abnormal_prob
                
                if is_abnormal:
                    abnormal_count += 1
                    # 높은 값 또는 낮은 값 선택
                    if random.random() < 0.7:  # 70%는 높은 값
                        value = round(normal_max * random.uniform(1.1, 2.0), 2)
                    else:  # 30%는 낮은 값
                        value = round(normal_min * random.uniform(0.5, 0.9), 2)
                    
                    flag = "H" if value > normal_max else "L"
                else:
                    # 정상 범위 내 값
                    value = round(random.uniform(normal_min, normal_max), 2)
                    flag = ""
                
                results[item] = {
                    "value": value,
                    "unit": test_info["unit"],
                    "normal_range": normal_range,
                    "flag": flag
                }
            
            # 의사 및 랩 정보
            doctor = random.choice(self.doctors.get(patient["department"], self.doctors["internal_medicine"]))
            
            # 메모 생성
            if abnormal_count == 0:
                interpretation = "모든 검사 결과가 정상 범위 내에 있습니다."
            elif abnormal_count == 1:
                interpretation = "한 항목에서 비정상 결과가 관찰됩니다. 추적 관찰이 필요할 수 있습니다."
            else:
                interpretation = f"{abnormal_count}개 항목에서 비정상 결과가 관찰됩니다. 추가 검사 및 평가가 필요합니다."
            
            lab_results.append({
                "date": test_date,
                "test_type": test_type,
                "ordering_doctor": doctor["name"],
                "ordering_doctor_id": doctor["id"],
                "lab_id": f"L{random.randint(10000, 99999)}",
                "results": results,
                "interpretation": interpretation,
                "collection_time": f"{random.randint(8, 17)}:{random.choice(['00', '15', '30', '45'])}",
                "report_time": f"{random.randint(9, 18)}:{random.choice(['00', '15', '30', '45'])}"
            })
        
        return lab_results

    def generate_imaging_studies(self, patient):
        """
        영상 검사 생성
        """
        # 환자당 0-3회의 영상 검사
        num_studies = random.randint(0, 3)
        imaging_studies = []
        
        # 각 영상 검사마다
        for _ in range(num_studies):
            # 검사일은 과거 날짜
            days_ago = random.randint(1, 365)  # 최대 1년 전
            study_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # 검사 유형 선택 (환자 상태에 따라)
            # medical_vector_db.py (계속)
            # 검사 유형 선택 (환자 상태에 따라)
            if "diagnoses" in patient and patient["diagnoses"]:
                diagnosis_names = [d["name"] for d in patient["diagnoses"]]
                
                if any(name in ["관상동맥질환", "심부전"] for name in diagnosis_names):
                    study_types = ["흉부 X선", "CT 흉부", "초음파 심장", "혈관조영술"]
                elif any(name in ["폐렴", "만성 폐쇄성 폐질환"] for name in diagnosis_names):
                    study_types = ["흉부 X선", "CT 흉부"]
                elif any(name in ["뇌졸중", "간질", "파킨슨병", "알츠하이머병"] for name in diagnosis_names):
                    study_types = ["CT 두부", "MRI 뇌"]
                elif any(name in ["급성 충수염", "담낭염", "장폐색", "대장 게실염"] for name in diagnosis_names):
                    study_types = ["복부 X선", "CT 복부", "초음파 복부"]
                else:
                    study_types = list(self.imaging_tests.keys())
            else:
                study_types = list(self.imaging_tests.keys())
            
            study_type = random.choice(study_types)
            
            # 의사 정보
            doctor = random.choice(self.doctors.get(patient["department"], self.doctors["internal_medicine"]))
            
            # 소견 생성
            if study_type in self.imaging_tests:
                possible_findings = self.imaging_tests[study_type]["common_findings"]
                
                # 진단에 따른 소견 추가
                if "diagnoses" in patient:
                    for diagnosis in patient["diagnoses"]:
                        if diagnosis["name"] == "관상동맥질환" and study_type in ["흉부 X선", "CT 흉부"]:
                            possible_findings.extend(["관상동맥 석회화", "심장 비대"])
                        elif diagnosis["name"] == "폐렴" and study_type in ["흉부 X선", "CT 흉부"]:
                            possible_findings.extend(["폐 경화", "간유리 음영"])
                        elif diagnosis["name"] == "뇌졸중" and study_type in ["CT 두부", "MRI 뇌"]:
                            possible_findings.extend(["뇌경색", "뇌출혈"])
                
                # 0-3개의 소견 선택
                num_findings = random.randint(0, 3)
                if num_findings > 0:
                    findings = random.sample(possible_findings, min(num_findings, len(possible_findings)))
                    finding_text = ", ".join(findings)
                    impression = f"{finding_text}가 관찰됩니다."
                else:
                    finding_text = "특이소견 없음"
                    impression = "정상 소견입니다."
            else:
                finding_text = "특이소견 없음"
                impression = "정상 소견입니다."
            
            imaging_studies.append({
                "date": study_date,
                "study_type": study_type,
                "ordering_doctor": doctor["name"],
                "ordering_doctor_id": doctor["id"],
                "radiologist": random.choice(["김영상", "박영상", "이영상"]),
                "study_id": f"I{random.randint(10000, 99999)}",
                "findings": finding_text,
                "impression": impression,
                "recommendation": random.choice([
                    "추가 검사 필요 없음",
                    "3개월 후 추적 검사 권장",
                    "6개월 후 추적 검사 권장",
                    "1년 후 추적 검사 권장",
                    "추가 검사(MRI) 권장",
                    "임상 상관관계 확인 필요"
                ])
            })
        
        return imaging_studies

    def generate_procedures(self, patient):
        """
        시술 및 수술 정보 생성
        """
        # 환자당 0-2회의 시술
        num_procedures = random.randint(0, 2)
        procedures_list = []
        
        # 각 시술마다
        for _ in range(num_procedures):
            # 시술일은 과거 날짜
            days_ago = random.randint(1, 365)  # 최대 1년 전
            procedure_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # 시술 유형 선택 (환자 상태에 따라)
            if "diagnoses" in patient and patient["diagnoses"]:
                diagnosis_names = [d["name"] for d in patient["diagnoses"]]
                
                suitable_procedures = []
                for proc_name, proc_info in self.procedures.items():
                    for diag in diagnosis_names:
                        if diag in proc_info.get("indications", []):
                            suitable_procedures.append(proc_name)
                
                if suitable_procedures:
                    procedure_name = random.choice(suitable_procedures)
                else:
                    procedure_name = random.choice(list(self.procedures.keys()))
            else:
                procedure_name = random.choice(list(self.procedures.keys()))
            
            # 시술 정보
            procedure_info = self.procedures.get(procedure_name, {})
            
            # 의사 정보
            doctor = random.choice(self.doctors.get("surgery", self.doctors["internal_medicine"]))
            
            # 시술 결과 및 합병증
            outcome = random.choice(["성공", "부분적 성공", "실패"])
            
            complications = []
            if random.random() < 0.2:  # 20% 확률로 합병증 발생
                potential_complications = procedure_info.get("complications", [])
                if potential_complications:
                    complications = random.sample(
                        potential_complications, 
                        min(random.randint(1, len(potential_complications)), len(potential_complications))
                    )
            
            procedures_list.append({
                "date": procedure_date,
                "name": procedure_name,
                "description": procedure_info.get("description", ""),
                "performing_doctor": doctor["name"],
                "performing_doctor_id": doctor["id"],
                "procedure_id": f"P{random.randint(10000, 99999)}",
                "location": random.choice(["수술실", "시술실", "내시경실", "중재시술실"]),
                "anesthesia": random.choice(["국소", "부분", "전신", "없음"]),
                "duration_minutes": random.randint(15, 240),
                "outcome": outcome,
                "complications": complications,
                "follow_up": random.choice([
                    "1주 후 외래 방문",
                    "2주 후 외래 방문",
                    "1개월 후 외래 방문",
                    "3개월 후 외래 방문",
                    "필요시 방문"
                ])
            })
        
        return procedures_list

    def generate_visits(self, patient, min_visits=1, max_visits=10):
        """
        진료 기록 생성
        """
        num_visits = random.randint(min_visits, max_visits)
        visits = []
        
        # 첫 방문은 가장 오래된 진단일보다 이전
        first_diagnosis_date = datetime.now()
        if "diagnoses" in patient and patient["diagnoses"]:
            for diagnosis in patient["diagnoses"]:
                diag_date = datetime.strptime(diagnosis["date"], "%Y-%m-%d")
                if diag_date < first_diagnosis_date:
                    first_diagnosis_date = diag_date
        
        # 첫 방문일은 첫 진단일보다 0-30일 이전
        first_visit_date = first_diagnosis_date - timedelta(days=random.randint(0, 30))
        
        # 방문 날짜 리스트 생성
        visit_dates = [first_visit_date]
        current_date = first_visit_date
        
        for _ in range(num_visits - 1):
            days_to_add = random.randint(14, 120)  # 2주에서 4개월 사이 간격
            current_date = current_date + timedelta(days=days_to_add)
            if current_date > datetime.now():
                break
            visit_dates.append(current_date)
        
        # 의사 선택 (진단에 있는 의사 우선, 없으면 무작위)
        doctors_from_diagnoses = set()
        if "diagnoses" in patient and patient["diagnoses"]:
            for diagnosis in patient["diagnoses"]:
                doctors_from_diagnoses.add((diagnosis["doctor"], diagnosis["doctor_id"]))
        
        # 각 방문에 대한 상세 정보 생성
        for i, visit_date in enumerate(visit_dates):
            # 의사 선택 (이전 방문 의사 유지 확률 높임)
            if i > 0 and random.random() < 0.7:  # 70% 확률로 이전 의사와 동일
                doctor_name = visits[i-1]["doctor"]
                doctor_id = visits[i-1]["doctor_id"]
            elif doctors_from_diagnoses and random.random() < 0.8:  # 80% 확률로 진단 의사 중 하나
                doctor_name, doctor_id = random.choice(list(doctors_from_diagnoses))
            else:
                doctor = random.choice(self.doctors.get(patient["department"], self.doctors["internal_medicine"]))
                doctor_name = doctor["name"]
                doctor_id = doctor["id"]
            
            # 활력징후
            vitals = self.generate_vitals(patient)
            
            # 방문 유형 및 이유
            if i == 0:
                visit_type = "초진"
                chief_complaint = random.choice([
                    "건강 검진", "두통", "복통", "어지러움", "기침", "발열", "무기력", 
                    "소화불량", "가슴 통증", "호흡곤란", "관절통", "혈뇨", "체중감소"
                ])
            else:
                visit_type = "재진"
                
                # 이전 진단에 따른 주요 증상
                if "diagnoses" in patient and patient["diagnoses"]:
                    diagnosis = random.choice(patient["diagnoses"])
                    disease = diagnosis["name"]
                    
                    if disease == "고혈압":
                        chief_complaint = random.choice([
                            "혈압 조절 확인", "두통", "어지러움", "약물 부작용 상담"
                        ])
                    elif disease == "당뇨병":
                        chief_complaint = random.choice([
                            "혈당 조절 확인", "다뇨", "다갈", "약물 부작용 상담", "발 저림"
                        ])
                    elif disease == "관상동맥질환":
                        chief_complaint = random.choice([
                            "가슴 통증", "호흡곤란", "약물 조절", "추적 관찰"
                        ])
                    elif disease in self.diagnosis_dict.get("surgery", []):
                        chief_complaint = random.choice([
                            "수술 후 상태 확인", "통증 재평가", "상처 치유 확인", "추적 관찰"
                        ])
                    else:
                        chief_complaint = random.choice([
                            "상태 확인", "약물 조절", "증상 재평가", "추적 관찰"
                        ])
                else:
                    chief_complaint = random.choice([
                        "정기 검진", "상태 확인", "약물 조절", "증상 재평가"
                    ])
            
            # 임상 노트
            subjective = self.generate_subjective_note(patient, chief_complaint)
            objective = self.generate_objective_note(patient, vitals)
            assessment = self.generate_assessment_note(patient)
            plan = self.generate_plan_note(patient)
            
            visits.append({
                "visit_id": f"V{i+1}-{patient['id']}",
                "date": visit_date.strftime("%Y-%m-%d"),
                "time": f"{random.randint(9, 17)}:{random.choice(['00', '15', '30', '45'])}",
                "type": visit_type,
                "department": patient["department"],
                "doctor": doctor_name,
                "doctor_id": doctor_id,
                "chief_complaint": chief_complaint,
                "vital_signs": vitals,
                "clinical_note": {
                    "subjective": subjective,
                    "objective": objective,
                    "assessment": assessment,
                    "plan": plan
                },
                "duration_minutes": random.randint(5, 30),
                "next_appointment": random.choice([
                    "1주 후", "2주 후", "1개월 후", "3개월 후", "6개월 후", "필요시 방문"
                ]) if random.random() < 0.8 else None
            })
        
        return visits

    def generate_vitals(self, patient):
        """
        활력징후 생성
        """
        # 나이에 따른 약간의 변동성 추가
        age = patient["age"]
        age_factor = age / 60  # 나이를 고려한 요소
        
        # 혈압 생성 (고령일수록 수축기 혈압이 높아지는 경향)
        systolic_base = 120 + (age_factor - 1) * 20
        diastolic_base = 80 + (age_factor - 1) * 5
        
        # 고혈압 진단이 있는 경우 추가 고려
        has_hypertension = False
        if "diagnoses" in patient:
            for diagnosis in patient["diagnoses"]:
                if "고혈압" in diagnosis["name"]:
                    has_hypertension = True
                    # 치료 전이라면 혈압이 더 높을 것
                    if diagnosis["status"] not in ["관해", "완치"]:
                        systolic_base += 20
                        diastolic_base += 10
        
        # 무작위성 추가
        systolic = max(90, min(200, round(systolic_base + random.uniform(-15, 15))))
        diastolic = max(60, min(110, round(diastolic_base + random.uniform(-10, 10))))
        
        vitals = {
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "pulse": round(random.uniform(60, 100)),
            "temperature": round(random.uniform(36.0, 37.5), 1),
            "respiratory_rate": round(random.uniform(12, 20)),
            "oxygen_saturation": round(random.uniform(95, 100))
        }
        
        # 당뇨 환자는 공복혈당 측정 추가
        if any(d["name"] == "당뇨병" for d in patient.get("diagnoses", [])):
            vitals["blood_glucose"] = round(random.uniform(100, 250))
        
        return vitals

    def generate_subjective_note(self, patient, chief_complaint):
        """
        주관적 노트 생성
        """
        # 기본 템플릿
        templates = [
            f"환자는 {chief_complaint}을(를) 호소합니다. 증상은 {random.choice(['가볍습니다', '중간 정도입니다', '심합니다'])}.",
            f"{chief_complaint} 관련 내원. 증상은 {random.choice(['최근 시작', '수일 전부터', '수주 전부터', '수개월 전부터'])} 있었다고 합니다.",
            f"주 호소: {chief_complaint}. 환자는 {random.choice(['경미한', '중등도의', '심한'])} 불편감을 보고합니다."
        ]
        
        note = random.choice(templates)
        
        # 추가 정보
        if random.random() < 0.7:  # 70% 확률로 추가 정보 포함
            additional_info = [
                f" 증상은 {random.choice(['휴식 시 완화됩니다', '움직일 때 악화됩니다', '식후 악화됩니다', '특별한 유발 요인이 없습니다'])}.",
                f" 환자는 {random.choice(['수면 장애', '피로감', '식욕 변화', '체중 변화'])}도 보고합니다.",
                f" 환자는 최근 {random.choice(['스트레스 증가', '식이 변화', '활동 수준 변화', '약물 변화'])}가 있었다고 합니다."
            ]
            note += random.choice(additional_info)
        
        # 진단 정보가 있는 경우, 관련 정보 추가
        if "diagnoses" in patient and patient["diagnoses"]:
            diagnosis = random.choice(patient["diagnoses"])
            diagnosis_note = f" 환자는 {diagnosis['date']}에 {diagnosis['name']}으로 진단받았습니다."
            
            if random.random() < 0.5:  # 50% 확률로 진단 정보 추가
                note += diagnosis_note
            
            # 약물 정보가 있는 경우
            if "medications" in patient and patient["medications"]:
                med_info = []
                for med in patient["medications"]:
                    if med["related_diagnosis"] == diagnosis["name"]:
                        med_info.append(f"{med['medication']} {med['dosage']} {med['frequency']}")
                
                if med_info and random.random() < 0.7:  # 70% 확률로 약물 정보 추가
                    note += f" 현재 {', '.join(med_info)}을(를) 복용 중입니다."
        
        return note

    def generate_objective_note(self, patient, vitals):
        """
        객관적 노트 생성
        """
        # 기본 템플릿
        note = f"혈압 {vitals['systolic_bp']}/{vitals['diastolic_bp']} mmHg, 맥박 {vitals['pulse']} bpm, 체온 {vitals['temperature']}°C, 호흡수 {vitals['respiratory_rate']}/분, 산소포화도 {vitals['oxygen_saturation']}%."
        
        # 신체 검진 결과 추가
        physical_exam = []
        
        # 진단에 따른 특이 소견 추가
        if "diagnoses" in patient and patient["diagnoses"]:
            for diagnosis in patient["diagnoses"]:
                if diagnosis["name"] == "고혈압":
                    if random.random() < 0.3:  # 30% 확률로 특이 소견
                        physical_exam.append("심음 청진 시 S4 갤럽 청진됨.")
                elif diagnosis["name"] == "당뇨병":
                    if random.random() < 0.4:  # 40% 확률로 특이 소견
                        physical_exam.append("발의 감각 저하 관찰됨.")
                elif diagnosis["name"] == "심부전":
                    if random.random() < 0.6:  # 60% 확률로 특이 소견
                        physical_exam.append("양측 하지 부종 관찰됨.")
                        physical_exam.append("폐 기저부에서 수포음 청진됨.")
                elif diagnosis["name"] == "만성 폐쇄성 폐질환":
                    if random.random() < 0.7:  # 70% 확률로 특이 소견
                        physical_exam.append("호기 시 천명음 청진됨.")
                        physical_exam.append("흉곽 확장 관찰됨.")
        
        # 일반 신체 검진 결과
        systems = [
            "심혈관계",
            "호흡기계",
            "소화기계",
            "신경계",
            "근골격계",
            "피부"
        ]
        
        # 1-3개의 시스템에 대한 검진 결과 추가
        num_systems = random.randint(1, 3)
        selected_systems = random.sample(systems, num_systems)
        
        for system in selected_systems:
            if system == "심혈관계":
                physical_exam.append(random.choice([
                    "규칙적인 심음, 심잡음 없음.",
                    "규칙적인 심음, 수축기 잡음 2/6 강도로 청진됨.",
                    "불규칙한 심음, 심잡음 없음."
                ]))
            elif system == "호흡기계":
                physical_exam.append(random.choice([
                    "폐 청진 정상, 수포음 없음.",
                    "양폐야에서 거친 호흡음 청진됨.",
                    "우측 폐에서 수포음 청진됨."
                ]))
            elif system == "소화기계":
                physical_exam.append(random.choice([
                    "복부 부드럽고 압통 없음.",
                    "경도의 상복부 압통 있음.",
                    "장음 정상, 간비종대 없음."
                ]))
            elif system == "신경계":
                physical_exam.append(random.choice([
                    "의식 명료, 뇌신경 기능 정상.",
                    "경도의 근력 약화 관찰됨.",
                    "감각 기능 정상."
                ]))
            elif system == "근골격계":
                physical_exam.append(random.choice([
                    "관절 운동 범위 정상.",
                    "경도의 관절 부종 관찰됨.",
                    "근력 5/5로 정상."
                ]))
            elif system == "피부":
                physical_exam.append(random.choice([
                    "피부 상태 양호, 발진 없음.",
                    "경도의 발진 관찰됨.",
                    "피부 탄력 정상."
                ]))
        
        if physical_exam:
            note += f" 신체 검진: {' '.join(physical_exam)}"
        else:
            note += " 신체 검진: 특이소견 없음."
        
        return note

    def generate_assessment_note(self, patient):
        """
        평가 노트 생성
        """
        if "diagnoses" not in patient or not patient["diagnoses"]:
            return "특이 진단 없음. 건강 상태 양호."
        
        assessments = []
        
        for diagnosis in patient["diagnoses"]:
            status = diagnosis["status"]
            severity = diagnosis["severity"]
            
            templates = [
                f"{diagnosis['name']}, {severity}, {status}.",
                f"{diagnosis['name']}: {severity} 수준, 현재 {status}.",
                f"{diagnosis['name']}({diagnosis['icd10']}): {status}."
            ]
            
            assessment = random.choice(templates)
            
            # 추가 평가 내용
            if random.random() < 0.5:  # 50% 확률로 추가 내용 포함
                if status == "활성":
                    assessment += random.choice([
                        " 증상 지속 중.",
                        " 약물 치료 중.",
                        " 추가 평가 필요."
                    ])
                elif status == "관해":
                    assessment += random.choice([
                        " 증상 호전됨.",
                        " 현재 치료에 좋은 반응 보임.",
                        " 안정적 상태 유지 중."
                    ])
                elif status == "완치":
                    assessment += random.choice([
                        " 추가 치료 필요 없음.",
                        " 정기 검진만 필요.",
                        " 재발 없음."
                    ])
            
            assessments.append(assessment)
        
        # 종합 평가
        if len(assessments) > 1:
            overall = random.choice([
                "복합 질환 관리 중.",
                "여러 질환의 상호작용 고려 필요.",
                "전반적 상태는 안정적."
            ])
            assessments.append(overall)
        
        return " ".join(assessments)

    def generate_plan_note(self, patient):
        """
        계획 노트 생성
        """
        plans = []
        
        # 약물 유지/조정
        if "medications" in patient and patient["medications"]:
            med_plan = random.choice([
                "현재 약물 유지.",
                "약물 용량 조절:",
                "약물 변경:",
                "추가 약물 고려:"
            ])
            
            if med_plan != "현재 약물 유지.":
                # 무작위로 약물 하나 선택
                med = random.choice(patient["medications"])
                med_plan += f" {med['medication']} {med['dosage']} {med['frequency']}."
            
            plans.append(med_plan)
        
        # 검사 계획
        if random.random() < 0.7:  # 70% 확률로 검사 계획 포함
            test_plans = [
                "다음 방문 시 기본 혈액 검사 시행.",
                "지질 프로필 검사 예정.",
                "HbA1c 측정 예정.",
                "간 기능 검사 시행.",
                "신장 기능 검사 예정.",
                "갑상선 기능 검사 고려.",
                "심전도 검사 예정.",
                "흉부 X선 촬영 예정.",
                "복부 초음파 검사 고려."
            ]
            
            plans.append(random.choice(test_plans))
        
        # 추적 관찰
        follow_up = random.choice([
            "1개월 후 재방문.",
            "3개월 후 재방문.",
            "6개월 후 재방문.",
            "증상 악화 시 조기 방문 권유."
        ])
        
        plans.append(follow_up)
        
        # 생활 습관 조언
        if random.random() < 0.5:  # 50% 확률로 생활 습관 조언 포함
            lifestyle_advice = random.choice([
                "저염식이 유지 권장.",
                "규칙적인 운동 권장.",
                "체중 관리 필요.",
                "금연 권고.",
                "알코올 섭취 제한 권고.",
                "충분한 수분 섭취 권장.",
                "스트레스 관리 권고."
            ])
            
            plans.append(lifestyle_advice)
        
        return " ".join(plans)

    def generate_complete_medical_record(self, department):
        """
        환자 전체 의무기록 생성
        """
        # 환자 기본 정보
        patient = self.generate_patient(department)
        
        # 진단 정보
        patient["diagnoses"] = self.generate_diagnoses(patient)
        
        # 약물 정보
        patient["medications"] = self.generate_medications(patient)
        
        # 검사 결과
        patient["lab_results"] = self.generate_lab_results(patient)
        
        # 영상 검사
        patient["imaging_studies"] = self.generate_imaging_studies(patient)
        
        # 시술 및 수술
        patient["procedures"] = self.generate_procedures(patient)
        
        # 진료 기록
        patient["visits"] = self.generate_visits(patient)
        
        return patient

    def generate_medical_dataset(self):
        """
        전체 의료 데이터셋 생성
        """
        dataset = {}
        
        for department, count in self.patient_counts.items():
            logger.info(f"{department} 환자 {count}명 생성 중...")
            
            patients = []
            for i in range(count):
                patient = self.generate_complete_medical_record(department)
                patients.append(patient)
            
            dataset[department] = patients
            
            # 파일로 저장
            output_path = self.output_dir / f"{department}_patients.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(patients, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{output_path}에 {len(patients)}명의 환자 데이터 저장 완료")
        
        return dataset


# medical_vector_db.py (계속)
class MedicalVectorStore:
    """
    의료 데이터를 위한 벡터 스토어 구축 클래스
    """
    def __init__(self, data_path="./medical_data", vector_store_path="./vector_stores"):
        """
        초기화 함수
        """
        self.data_path = Path(data_path)
        self.vector_store_path = Path(vector_store_path)
        
        # 기본 디렉토리 생성
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # 한국어에 최적화된 임베딩 모델 사용
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask"
        )
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # 문서 분할기 설정 - 의료 문서에 적합하게 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def load_medical_data(self, file_pattern="*_patients.json"):
        """
        의료 데이터 로드
        """
        import glob
        
        data_files = list(self.data_path.glob(file_pattern))
        
        if not data_files:
            logger.warning(f"No files matching {file_pattern} found in {self.data_path}")
            return []
        
        documents = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    patients = json.load(f)
                
                department = file_path.stem.replace("_patients", "")
                logger.info(f"Loading {len(patients)} patients from {department} department")
                
                # 각 환자 정보를 문서로 변환
                for patient in patients:
                    documents.extend(self._convert_patient_to_documents(patient, department))
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} total documents from medical data")
        return documents
    
    def _convert_patient_to_documents(self, patient, department):
        """
        환자 정보를 여러 개의 문서로 변환 (세분화된 정보)
        """
        from langchain.schema import Document
        
        documents = []
        
        # 환자 기본 정보 문서
        basic_info = f"""
        환자 ID: {patient['id']}
        이름: {patient['name']}
        성별: {patient['gender']}
        나이: {patient['age']}
        생년월일: {patient['birthdate']}
        혈액형: {patient.get('blood_type', '정보 없음')}
        키: {patient.get('height', '정보 없음')} cm
        체중: {patient.get('weight', '정보 없음')} kg
        BMI: {patient.get('bmi', '정보 없음')}
        주소: {patient.get('address', '정보 없음')}
        전화번호: {patient.get('phone', '정보 없음')}
        보험: {patient.get('insurance', '정보 없음')}
        진료과: {department}
        """
        
        if patient.get('allergies'):
            basic_info += f"\n알레르기: {', '.join(patient['allergies'])}"
        
        if patient.get('smoking'):
            basic_info += f"\n흡연: {patient['smoking']['status']}"
            if patient['smoking'].get('details'):
                basic_info += f" ({patient['smoking']['details']})"
        
        if patient.get('alcohol'):
            basic_info += f"\n음주: {patient['alcohol']['status']}"
            if patient['alcohol'].get('details'):
                basic_info += f" ({patient['alcohol']['details']})"
        
        documents.append(Document(
            page_content=basic_info.strip(),
            metadata={
                "patient_id": patient['id'],
                "name": patient['name'],
                "gender": patient['gender'],
                "age": patient['age'],
                "department": department,
                "document_type": "basic_info"
            }
        ))
        
        # 진단 정보 문서
        if patient.get('diagnoses'):
            for i, diagnosis in enumerate(patient['diagnoses']):
                diagnosis_doc = f"""
                환자 ID: {patient['id']}
                이름: {patient['name']}
                성별: {patient['gender']}
                나이: {patient['age']}
                
                [진단 정보 {i+1}]
                진단명: {diagnosis['name']}
                ICD10 코드: {diagnosis.get('icd10', '정보 없음')}
                진단일: {diagnosis.get('date', '정보 없음')}
                진단 의사: {diagnosis.get('doctor', '정보 없음')} (ID: {diagnosis.get('doctor_id', '정보 없음')})
                확신도: {diagnosis.get('confidence', '정보 없음')}
                상태: {diagnosis.get('status', '정보 없음')}
                중증도: {diagnosis.get('severity', '정보 없음')}
                메모: {diagnosis.get('memo', '정보 없음')}
                증상: {', '.join(diagnosis.get('symptoms', ['정보 없음']))}
                """
                
                documents.append(Document(
                    page_content=diagnosis_doc.strip(),
                    metadata={
                        "patient_id": patient['id'],
                        "name": patient['name'],
                        "gender": patient['gender'],
                        "age": patient['age'],
                        "department": department,
                        "document_type": "diagnosis",
                        "diagnosis_name": diagnosis['name'],
                        "diagnosis_date": diagnosis.get('date', ''),
                        "diagnosis_status": diagnosis.get('status', '')
                    }
                ))
        
        # 약물 정보 문서
        if patient.get('medications'):
            for i, medication in enumerate(patient['medications']):
                medication_doc = f"""
                환자 ID: {patient['id']}
                이름: {patient['name']}
                성별: {patient['gender']}
                나이: {patient['age']}
                
                [약물 정보 {i+1}]
                약물명: {medication['medication']}
                약물 분류: {medication.get('class', '정보 없음')}
                처방일: {medication.get('prescription_date', '정보 없음')}
                처방 기간: {medication.get('duration_days', '정보 없음')}일
                용량: {medication.get('dosage', '정보 없음')}
                빈도: {medication.get('frequency', '정보 없음')}
                재처방 횟수: {medication.get('refill', '정보 없음')}
                처방 의사: {medication.get('doctor', '정보 없음')} (ID: {medication.get('doctor_id', '정보 없음')})
                관련 진단: {medication.get('related_diagnosis', '정보 없음')}
                특별 지시사항: {medication.get('special_instructions', '정보 없음')}
                """
                
                documents.append(Document(
                    page_content=medication_doc.strip(),
                    metadata={
                        "patient_id": patient['id'],
                        "name": patient['name'],
                        "gender": patient['gender'],
                        "age": patient['age'],
                        "department": department,
                        "document_type": "medication",
                        "medication_name": medication['medication'],
                        "medication_class": medication.get('class', ''),
                        "related_diagnosis": medication.get('related_diagnosis', '')
                    }
                ))
        
        # 검사 결과 문서
        if patient.get('lab_results'):
            for i, lab in enumerate(patient['lab_results']):
                lab_doc = f"""
                환자 ID: {patient['id']}
                이름: {patient['name']}
                성별: {patient['gender']}
                나이: {patient['age']}
                
                [검사 결과 {i+1}]
                검사일: {lab.get('date', '정보 없음')}
                검사 유형: {lab.get('test_type', '정보 없음')}
                검사 요청 의사: {lab.get('ordering_doctor', '정보 없음')} (ID: {lab.get('ordering_doctor_id', '정보 없음')})
                검사 ID: {lab.get('lab_id', '정보 없음')}
                검체 채취 시간: {lab.get('collection_time', '정보 없음')}
                보고 시간: {lab.get('report_time', '정보 없음')}
                
                결과 항목:
                """
                
                for test_name, test_result in lab.get('results', {}).items():
                    lab_doc += f"""
                    - {test_name}: {test_result.get('value', '정보 없음')} {test_result.get('unit', '')} 
                      (정상 범위: {test_result.get('normal_range', '정보 없음')})
                      {test_result.get('flag', '')}
                    """
                
                if lab.get('interpretation'):
                    lab_doc += f"\n해석: {lab['interpretation']}"
                
                documents.append(Document(
                    page_content=lab_doc.strip(),
                    metadata={
                        "patient_id": patient['id'],
                        "name": patient['name'],
                        "gender": patient['gender'],
                        "age": patient['age'],
                        "department": department,
                        "document_type": "lab_result",
                        "lab_date": lab.get('date', ''),
                        "test_type": lab.get('test_type', '')
                    }
                ))
        
        # 영상 검사 문서
        if patient.get('imaging_studies'):
            for i, study in enumerate(patient['imaging_studies']):
                imaging_doc = f"""
                환자 ID: {patient['id']}
                이름: {patient['name']}
                성별: {patient['gender']}
                나이: {patient['age']}
                
                [영상 검사 {i+1}]
                검사일: {study.get('date', '정보 없음')}
                검사 유형: {study.get('study_type', '정보 없음')}
                검사 요청 의사: {study.get('ordering_doctor', '정보 없음')} (ID: {study.get('ordering_doctor_id', '정보 없음')})
                영상의학과 의사: {study.get('radiologist', '정보 없음')}
                검사 ID: {study.get('study_id', '정보 없음')}
                
                소견: {study.get('findings', '정보 없음')}
                판독: {study.get('impression', '정보 없음')}
                추천: {study.get('recommendation', '정보 없음')}
                """
                
                documents.append(Document(
                    page_content=imaging_doc.strip(),
                    metadata={
                        "patient_id": patient['id'],
                        "name": patient['name'],
                        "gender": patient['gender'],
                        "age": patient['age'],
                        "department": department,
                        "document_type": "imaging_study",
                        "study_date": study.get('date', ''),
                        "study_type": study.get('study_type', '')
                    }
                ))
        
        # 시술 및 수술 문서
        if patient.get('procedures'):
            for i, procedure in enumerate(patient['procedures']):
                procedure_doc = f"""
                환자 ID: {patient['id']}
                이름: {patient['name']}
                성별: {patient['gender']}
                나이: {patient['age']}
                
                [시술/수술 {i+1}]
                시술일: {procedure.get('date', '정보 없음')}
                시술명: {procedure.get('name', '정보 없음')}
                설명: {procedure.get('description', '정보 없음')}
                시술 의사: {procedure.get('performing_doctor', '정보 없음')} (ID: {procedure.get('performing_doctor_id', '정보 없음')})
                시술 ID: {procedure.get('procedure_id', '정보 없음')}
                위치: {procedure.get('location', '정보 없음')}
                마취: {procedure.get('anesthesia', '정보 없음')}
                소요 시간: {procedure.get('duration_minutes', '정보 없음')}분
                결과: {procedure.get('outcome', '정보 없음')}
                """
                
                if procedure.get('complications'):
                    procedure_doc += f"\n합병증: {', '.join(procedure['complications'])}"
                
                procedure_doc += f"\n추적 관찰: {procedure.get('follow_up', '정보 없음')}"
                
                documents.append(Document(
                    page_content=procedure_doc.strip(),
                    metadata={
                        "patient_id": patient['id'],
                        "name": patient['name'],
                        "gender": patient['gender'],
                        "age": patient['age'],
                        "department": department,
                        "document_type": "procedure",
                        "procedure_date": procedure.get('date', ''),
                        "procedure_name": procedure.get('name', ''),
                        "outcome": procedure.get('outcome', '')
                    }
                ))
        
        # 진료 기록 문서
        if patient.get('visits'):
            for i, visit in enumerate(patient['visits']):
                visit_doc = f"""
                환자 ID: {patient['id']}
                이름: {patient['name']}
                성별: {patient['gender']}
                나이: {patient['age']}
                
                [진료 기록 {i+1}]
                방문 ID: {visit.get('visit_id', '정보 없음')}
                방문일: {visit.get('date', '정보 없음')}
                방문 시간: {visit.get('time', '정보 없음')}
                방문 유형: {visit.get('type', '정보 없음')}
                진료과: {visit.get('department', '정보 없음')}
                담당 의사: {visit.get('doctor', '정보 없음')} (ID: {visit.get('doctor_id', '정보 없음')})
                주 호소: {visit.get('chief_complaint', '정보 없음')}
                
                활력 징후:
                수축기 혈압: {visit.get('vital_signs', {}).get('systolic_bp', '정보 없음')} mmHg
                이완기 혈압: {visit.get('vital_signs', {}).get('diastolic_bp', '정보 없음')} mmHg
                맥박: {visit.get('vital_signs', {}).get('pulse', '정보 없음')} bpm
                체온: {visit.get('vital_signs', {}).get('temperature', '정보 없음')} °C
                호흡수: {visit.get('vital_signs', {}).get('respiratory_rate', '정보 없음')} /분
                산소포화도: {visit.get('vital_signs', {}).get('oxygen_saturation', '정보 없음')} %
                """
                
                if 'blood_glucose' in visit.get('vital_signs', {}):
                    visit_doc += f"혈당: {visit['vital_signs']['blood_glucose']} mg/dL\n"
                
                visit_doc += f"""
                임상 노트:
                주관적(S): {visit.get('clinical_note', {}).get('subjective', '정보 없음')}
                객관적(O): {visit.get('clinical_note', {}).get('objective', '정보 없음')}
                평가(A): {visit.get('clinical_note', {}).get('assessment', '정보 없음')}
                계획(P): {visit.get('clinical_note', {}).get('plan', '정보 없음')}
                
                진료 시간: {visit.get('duration_minutes', '정보 없음')}분
                """
                
                if visit.get('next_appointment'):
                    visit_doc += f"다음 예약: {visit['next_appointment']}"
                
                documents.append(Document(
                    page_content=visit_doc.strip(),
                    metadata={
                        "patient_id": patient['id'],
                        "name": patient['name'],
                        "gender": patient['gender'],
                        "age": patient['age'],
                        "department": department,
                        "document_type": "visit",
                        "visit_date": visit.get('date', ''),
                        "visit_type": visit.get('type', ''),
                        "chief_complaint": visit.get('chief_complaint', '')
                    }
                ))
        
        # 통합 문서 (전체 환자 기록을 하나의 문서로)
        # 이는 큰 컨텍스트가 필요한 질문에 유용함
        integrated_doc = f"""
        [환자 통합 기록]
        환자 ID: {patient['id']}
        이름: {patient['name']}
        성별: {patient['gender']}
        나이: {patient['age']}
        생년월일: {patient['birthdate']}
        진료과: {department}
        
        [진단 요약]
        """
        
        if patient.get('diagnoses'):
            for diagnosis in patient['diagnoses']:
                integrated_doc += f"""
                - {diagnosis['name']} ({diagnosis.get('date', '날짜 없음')})
                  상태: {diagnosis.get('status', '정보 없음')}, 중증도: {diagnosis.get('severity', '정보 없음')}
                """
        else:
            integrated_doc += "진단 정보 없음\n"
        
        integrated_doc += "\n[약물 요약]\n"
        if patient.get('medications'):
            for med in patient['medications']:
                integrated_doc += f"""
                - {med['medication']} {med.get('dosage', '')} {med.get('frequency', '')}
                  처방일: {med.get('prescription_date', '정보 없음')}, 관련 진단: {med.get('related_diagnosis', '정보 없음')}
                """
        else:
            integrated_doc += "약물 정보 없음\n"
        
        integrated_doc += "\n[최근 검사 결과 요약]\n"
        if patient.get('lab_results'):
            # 가장 최근 검사 결과만 포함
            recent_lab = max(patient['lab_results'], key=lambda x: x.get('date', ''))
            integrated_doc += f"검사일: {recent_lab.get('date', '정보 없음')}, 검사 유형: {recent_lab.get('test_type', '정보 없음')}\n"
            
            for test_name, test_result in recent_lab.get('results', {}).items():
                flag = test_result.get('flag', '')
                if flag:
                    integrated_doc += f"- {test_name}: {test_result.get('value', '')} {test_result.get('unit', '')} ({flag})\n"
        else:
            integrated_doc += "검사 결과 정보 없음\n"
        
        integrated_doc += "\n[최근 방문 요약]\n"
        if patient.get('visits'):
            # 가장 최근 방문만 포함
            recent_visit = max(patient['visits'], key=lambda x: x.get('date', ''))
            integrated_doc += f"""
            방문일: {recent_visit.get('date', '정보 없음')}
            주 호소: {recent_visit.get('chief_complaint', '정보 없음')}
            평가: {recent_visit.get('clinical_note', {}).get('assessment', '정보 없음')}
            계획: {recent_visit.get('clinical_note', {}).get('plan', '정보 없음')}
            """
        else:
            integrated_doc += "방문 기록 없음\n"
        
        documents.append(Document(
            page_content=integrated_doc.strip(),
            metadata={
                "patient_id": patient['id'],
                "name": patient['name'],
                "gender": patient['gender'],
                "age": patient['age'],
                "department": department,
                "document_type": "integrated_record"
            }
        ))
        
        return documents
    
    def create_vector_store(self, documents, store_name="medical_vector_store"):
        """
        벡터 스토어 생성
        """
        if not documents:
            logger.warning("벡터 스토어를 생성할 문서가 없습니다.")
            return None
        
        logger.info(f"{len(documents)}개 문서로 벡터 스토어 생성 중...")
        
        # 문서를 청크로 분할
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"총 {len(chunks)}개의 청크 생성")
        
        from langchain_community.vectorstores import FAISS
        
        # 벡터 스토어 경로
        store_path = self.vector_store_path / store_name
        store_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 벡터 스토어 생성
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(store_path)
        
        logger.info(f"벡터 스토어가 {store_path}에 저장되었습니다.")
        return vectorstore
    
    def load_vector_store(self, store_name="medical_vector_store"):
        """
        저장된 벡터 스토어 로드
        """
        from langchain_community.vectorstores import FAISS
        
        store_path = self.vector_store_path / store_name
        
        if not store_path.exists():
            logger.error(f"벡터 스토어 경로가 존재하지 않습니다: {store_path}")
            return None
        
        logger.info(f"{store_path}에서 벡터 스토어 로드 중...")
        
        try:
            # allow_dangerous_deserialization=True 옵션 추가
            vectorstore = FAISS.load_local(
                store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("벡터 스토어 로드 완료")
            return vectorstore
        except Exception as e:
            logger.error(f"벡터 스토어 로드 중 오류 발생: {e}")
            return None
    
    def search_similar_documents(self, query, vectorstore, k=5, filter_dict=None):
        """
        유사 문서 검색 (메타데이터 필터링 지원)
        """
        if not vectorstore:
            logger.error("유효한 벡터 스토어가 없습니다.")
            return []
        
        logger.info(f"쿼리로 검색 중: {query}")
        
        if filter_dict:
            # 메타데이터 필터 적용한 검색
            docs = vectorstore.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            # 기본 유사도 검색
            docs = vectorstore.similarity_search(query, k=k)
        
        return docs
    
    def search_hybrid(self, query, vectorstore, k=5, filter_dict=None):
        """
        하이브리드 검색 (유사도 + 키워드)
        """
        if not vectorstore:
            logger.error("유효한 벡터 스토어가 없습니다.")
            return []
        
        # 키워드 검색을 위한 인덱스 생성
        from langchain.retrievers import BM25Retriever
        
        # BM25 검색기 생성
        all_documents = vectorstore.docstore._dict.values()
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = k
        
        # 두 검색 결과 결합
        vector_results = set(doc.page_content for doc in self.search_similar_documents(query, vectorstore, k, filter_dict))
        keyword_results = set(doc.page_content for doc in bm25_retriever.get_relevant_documents(query))
        
        # 두 결과 모두에 있는 문서 우선
        combined_results = []
        
        # 두 결과 모두에 있는 문서
        for doc in all_documents:
            if doc.page_content in vector_results and doc.page_content in keyword_results:
                combined_results.append(doc)
        
        # 벡터 검색 결과만 있는 문서
        for doc in all_documents:
            if doc.page_content in vector_results and doc.page_content not in keyword_results:
                if len(combined_results) < k:
                    combined_results.append(doc)
        
        # 키워드 검색 결과만 있는 문서
        for doc in all_documents:
            if doc.page_content not in vector_results and doc.page_content in keyword_results:
                if len(combined_results) < k:
                    combined_results.append(doc)
        
        return combined_results[:k]
    
    def advanced_medical_search(self, query, vectorstore, age_filter=None, gender=None, department=None, 
                              diagnosis=None, date_range=None, document_type=None, k=5):
        """
        고급 의료 검색 (다양한 필터 조합)
        """
        # 필터 딕셔너리 구성
        filter_dict = {}
        
        if age_filter:
            min_age, max_age = age_filter
            filter_dict["age"] = {"$gte": min_age, "$lte": max_age}
        
        if gender:
            filter_dict["gender"] = gender
        
        if department:
            filter_dict["department"] = department
        
        if diagnosis:
            filter_dict["diagnosis_name"] = diagnosis
        
        if document_type:
            filter_dict["document_type"] = document_type
        
        # 다양한 필터 조합을 적용한 검색
        docs = self.search_similar_documents(query, vectorstore, k, filter_dict)
        
        # 후처리: 날짜 필터링 (벡터 스토어의 기본 기능에 없는 경우)
        if date_range and docs:
            from datetime import datetime
            
            start_date, end_date = date_range
            filtered_docs = []
            
            for doc in docs:
                # 문서 유형에 따라 날짜 필드 선택
                date_field = None
                if doc.metadata.get("document_type") == "visit":
                    date_field = "visit_date"
                elif doc.metadata.get("document_type") == "diagnosis":
                    date_field = "diagnosis_date"
                elif doc.metadata.get("document_type") == "lab_result":
                    date_field = "lab_date"
                elif doc.metadata.get("document_type") == "procedure":
                    date_field = "procedure_date"
                
                # 날짜 필드가 있고 값이 있으면 필터링
                if date_field and doc.metadata.get(date_field):
                    try:
                        doc_date = datetime.strptime(doc.metadata[date_field], "%Y-%m-%d")
                        if start_date <= doc_date <= end_date:
                            filtered_docs.append(doc)
                    except:
                        pass
                else:
                    # 날짜 필드가 없으면 포함 (통합 문서 등)
                    filtered_docs.append(doc)
            
            return filtered_docs
        
        return docs
    
    # medical_vector_db.py (계속)
    def semantic_medical_query_expansion(self, original_query):
        """
        의료 용어 확장을 통한 검색 질의 개선
        """
        # 의료 용어 사전 (예시)
        medical_terms = {
            "고혈압": ["hypertension", "혈압 상승", "혈압 조절 문제", "고혈압성 질환"],
            "당뇨": ["당뇨병", "diabetes", "혈당 조절 장애", "인슐린 저항성", "제2형 당뇨병"],
            "심장병": ["심장 질환", "관상동맥질환", "심근경색", "협심증", "심부전"],
            "뇌졸중": ["stroke", "CVA", "뇌경색", "뇌출혈", "대뇌 혈관 사고"],
            "폐렴": ["pneumonia", "폐 감염", "하부 호흡기 감염"],
            "위염": ["gastritis", "위장염", "소화불량", "위산 역류"],
            "수술": ["외과적 치료", "시술", "절제술", "수술적 치료"],
            "임신": ["pregnancy", "임신성", "모성", "태아"],
            "알레르기": ["allergy", "과민 반응", "과민성", "아토피"],
            "두통": ["headache", "편두통", "긴장성 두통", "군발성 두통"],
        }
        
        # 약물 그룹
        medication_groups = {
            "항고혈압제": ["ACE 억제제", "ARB", "베타차단제", "칼슘채널차단제", "이뇨제", 
                      "암로디핀", "로자탄", "에날라프릴", "히드로클로로티아지드", "메토프롤롤"],
            "혈당강하제": ["메트포르민", "설포닐우레아", "DPP-4 억제제", "SGLT2 억제제", "인슐린",
                      "글리메피리드", "리나글립틴", "엠파글리플로진"],
            "지질강하제": ["스타틴", "아토르바스타틴", "로수바스타틴", "심바스타틴", "에제티미브"],
            "항혈소판제": ["아스피린", "클로피도그렐", "티카그렐러"],
            "항응고제": ["와파린", "리바록사반", "아픽사반", "다비가트란"],
            "진통제": ["아세트아미노펜", "이부프로펜", "나프록센", "트라마돌", "모르핀"]
        }
        
        # 검사 그룹
        test_groups = {
            "혈액검사": ["CBC", "전혈구검사", "혈색소", "백혈구", "혈소판", "적혈구"],
            "간기능검사": ["LFT", "AST", "ALT", "ALP", "GGT", "빌리루빈"],
            "신장기능검사": ["BUN", "크레아티닌", "eGFR", "요산"],
            "지질검사": ["콜레스테롤", "LDL", "HDL", "중성지방", "지질 프로필"],
            "당뇨검사": ["공복혈당", "당화혈색소", "HbA1c", "인슐린", "OGTT"],
            "심장표지자": ["트로포닌", "CK-MB", "BNP", "NT-proBNP"],
            "갑상선검사": ["TSH", "Free T4", "T3"]
        }
        
        # 증상 그룹
        symptom_groups = {
            "흉부증상": ["가슴통증", "흉통", "가슴 불편감", "심계항진", "호흡곤란"],
            "위장증상": ["복통", "소화불량", "구역", "구토", "설사", "변비", "복부 팽만"],
            "신경계증상": ["두통", "어지러움", "현기증", "마비", "저림", "감각 이상", "경련"],
            "호흡기증상": ["기침", "가래", "천명음", "호흡곤란", "콧물", "재채기", "인후통"],
            "피부증상": ["발진", "가려움", "두드러기", "부종", "홍반", "탈모"],
            "근골격계증상": ["관절통", "근육통", "요통", "경부통", "강직", "부종"]
        }
        
        # 원본 쿼리 복사
        expanded_query = original_query
        
        # 의료 용어 확장
        for term, synonyms in medical_terms.items():
            if term in original_query:
                synonyms_str = " OR ".join([f'"{s}"' for s in synonyms if s not in original_query])
                if synonyms_str:
                    expanded_query += f" OR ({synonyms_str})"
        
        # 약물 그룹 확장
        for group, meds in medication_groups.items():
            if group in original_query:
                meds_str = " OR ".join([f'"{m}"' for m in meds if m not in original_query])
                if meds_str:
                    expanded_query += f" OR ({meds_str})"
        
        # 검사 그룹 확장
        for group, tests in test_groups.items():
            if group in original_query:
                tests_str = " OR ".join([f'"{t}"' for t in tests if t not in original_query])
                if tests_str:
                    expanded_query += f" OR ({tests_str})"
        
        # 증상 그룹 확장
        for group, symptoms in symptom_groups.items():
            if group in original_query:
                symptoms_str = " OR ".join([f'"{s}"' for s in symptoms if s not in original_query])
                if symptoms_str:
                    expanded_query += f" OR ({symptoms_str})"
        
        logger.info(f"원본 쿼리: {original_query}")
        logger.info(f"확장 쿼리: {expanded_query}")
        
        return expanded_query
    
    def build_qa_chain(self, vectorstore):
        """
        질의응답 체인 구축 (RetrievalQA)
        """
        try:
            from langchain.chains import RetrievalQA
            from langchain_community.llms import HuggingFacePipeline
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            # 한국어 LLM 로드
            model_name = "beomi/KoAlpaca-Polyglot-5.8B"  # 또는 다른 한국어 모델
            
            logger.info(f"LLM 모델 {model_name} 로딩 중...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # 검색기 설정
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # QA 체인 구축
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            return qa
        
        except Exception as e:
            logger.error(f"QA 체인 구축 중 오류 발생: {e}")
            return None
    
    def clinical_case_search(self, vectorstore, patient_description):
        """
        환자 사례 설명을 기반으로 유사 임상 사례 검색
        """
        # 예시 사용법:
        # patient_description = """
        # 62세 남성 환자, 고혈압과 당뇨병 병력 있음. 
        # 최근 흉통, 호흡곤란, 발한 증상으로 응급실 내원. 
        # 심전도에서 ST 분절 상승 관찰됨.
        # """
        
        # 환자 특성 추출 (간단한 규칙 기반)
        age = None
        gender = None
        diagnosis = []
        symptoms = []
        
        # 연령 추출
        import re
        age_match = re.search(r'(\d+)세', patient_description)
        if age_match:
            age = int(age_match.group(1))
        
        # 성별 추출
        if '남성' in patient_description or '남자' in patient_description:
            gender = '남'
        elif '여성' in patient_description or '여자' in patient_description:
            gender = '여'
        
        # 진단명 추출 (간단한 매칭)
        common_diagnoses = [
            "고혈압", "당뇨병", "고지혈증", "심부전", "관상동맥질환", "뇌졸중", 
            "간경변", "신부전", "폐렴", "천식", "만성폐쇄성폐질환", "위염", "위궤양"
        ]
        
        for d in common_diagnoses:
            if d in patient_description:
                diagnosis.append(d)
        
        # 증상 추출 (간단한 매칭)
        common_symptoms = [
            "흉통", "호흡곤란", "복통", "두통", "어지러움", "구역", "구토", 
            "설사", "발열", "오한", "발한", "피로", "무기력", "체중감소"
        ]
        
        for s in common_symptoms:
            if s in patient_description:
                symptoms.append(s)
        
        logger.info(f"추출된 환자 특성: 나이={age}, 성별={gender}, 진단={diagnosis}, 증상={symptoms}")
        
        # 필터 구성
        filter_dict = {}
        
        if age:
            min_age = max(20, age - 10)
            max_age = min(90, age + 10)
            filter_dict["age"] = {"$gte": min_age, "$lte": max_age}
        
        if gender:
            filter_dict["gender"] = gender
        
        # 유사 사례 검색 (진단명 또는 증상 기반)
        search_terms = []
        if diagnosis:
            search_terms.extend(diagnosis)
        if symptoms:
            search_terms.extend(symptoms)
        
        if not search_terms:
            search_terms = [patient_description]
        
        query = " ".join(search_terms)
        
        # 메타데이터 필터를 적용한 검색
        # 통합 기록 문서와 진단 문서만 검색
        if "document_type" not in filter_dict:
            filter_dict["document_type"] = {"$in": ["integrated_record", "diagnosis"]}
        
        results = self.search_similar_documents(query, vectorstore, k=5, filter_dict=filter_dict)
        
        return results
    
    def create_vector_indices(self):
        """
        다양한 인덱스 및 벡터 스토어 구축
        """
        # 전체 의료 데이터 로드
        all_documents = self.load_medical_data()
        
        if not all_documents:
            logger.warning("벡터 인덱스를 생성할 문서가 없습니다.")
            return {}
        
        # 다양한 인덱스 구축
        indices = {}
        
        # 1. 전체 통합 인덱스
        logger.info("전체 통합 인덱스 생성 중...")
        indices["general"] = self.create_vector_store(all_documents, "general_index")
        
        # 2. 진단별 인덱스
        logger.info("진단별 인덱스 생성 중...")
        diagnosis_docs = [doc for doc in all_documents if doc.metadata.get("document_type") == "diagnosis"]
        if diagnosis_docs:
            indices["diagnosis"] = self.create_vector_store(diagnosis_docs, "diagnosis_index")
        
        # 3. 약물별 인덱스
        logger.info("약물별 인덱스 생성 중...")
        medication_docs = [doc for doc in all_documents if doc.metadata.get("document_type") == "medication"]
        if medication_docs:
            indices["medication"] = self.create_vector_store(medication_docs, "medication_index")
        
        # 4. 검사 결과 인덱스
        logger.info("검사 결과 인덱스 생성 중...")
        lab_docs = [doc for doc in all_documents if doc.metadata.get("document_type") == "lab_result"]
        if lab_docs:
            indices["lab_results"] = self.create_vector_store(lab_docs, "lab_results_index")
        
        # 5. 진료 기록 인덱스
        logger.info("진료 기록 인덱스 생성 중...")
        visit_docs = [doc for doc in all_documents if doc.metadata.get("document_type") == "visit"]
        if visit_docs:
            indices["visits"] = self.create_vector_store(visit_docs, "visits_index")
        
        # 6. 진료과별 인덱스
        logger.info("진료과별 인덱스 생성 중...")
        
        departments = set(doc.metadata.get("department", "") for doc in all_documents)
        
        for dept in departments:
            if dept:
                dept_docs = [doc for doc in all_documents if doc.metadata.get("department") == dept]
                if dept_docs:
                    indices[f"dept_{dept}"] = self.create_vector_store(
                        dept_docs, 
                        f"department_{dept}_index"
                    )
        
        return indices


# 메인 실행 함수
def main():
    # # 1. 의료 데이터 생성
    # logger.info("의료 데이터 생성 중...")
    # data_generator = MedicalDataGenerator()
    # dataset = data_generator.generate_medical_dataset() =>이미 생성해서 data에 올라가있습니다
    
    # 2. 벡터 스토어 구축
    # logger.info("벡터 스토어 구축 중...")
    # vector_store = MedicalVectorStore()
    
    # # 3. 모든 문서 로드
    # documents = vector_store.load_medical_data()
    
    # # 4. 통합 벡터 스토어 생성
    # vectorstore = vector_store.create_vector_store(documents)
    

    # 벡터 스토어 객체 초기화
    vs_builder = MedicalVectorStore()
    
    # 기존 벡터 스토어 로드
    logger.info("벡터 스토어 로드 중...")
    vectorstore = vs_builder.load_vector_store("medical_vector_store")
    
    if vectorstore:
        # 기본 검색
        query = "고혈압 환자의 최근 혈압 측정 기록"
        # 여기서 변수명 수정 (vector_store -> vs_builder)
        results = vs_builder.search_similar_documents(query, vectorstore, k=3)
        
        print(f"\n기본 검색 쿼리: {query}")
        print(f"검색 결과 ({len(results)}개):")
        for i, doc in enumerate(results, 1):
            print(f"\n결과 {i}:")
            print(f"유형: {doc.metadata.get('document_type', '알 수 없음')}")
            print(f"환자 ID: {doc.metadata.get('patient_id', '알 수 없음')}")
            print(f"내용 미리보기: {doc.page_content[:200]}...")
            print("-" * 50)
        
        # 고급 검색
        query = "60세 이상 남성 환자 중 심장 질환과 당뇨병을 동시에 가진 환자"
        # 여기서도 변수명 수정 (vector_store -> vs_builder)
        advanced_results = vs_builder.advanced_medical_search(
            query, 
            vectorstore, 
            age_filter=(60, 100), 
            gender="남",
            document_type="integrated_record",
            k=3
        )
        
        print(f"\n고급 검색 쿼리: {query}")
        print(f"검색 결과 ({len(advanced_results)}개):")
        for i, doc in enumerate(advanced_results, 1):
            print(f"\n결과 {i}:")
            print(f"환자 ID: {doc.metadata.get('patient_id', '알 수 없음')}")
            print(f"나이: {doc.metadata.get('age', '알 수 없음')}")
            print(f"성별: {doc.metadata.get('gender', '알 수 없음')}")
            print(f"내용 미리보기: {doc.page_content[:200]}...")
            print("-" * 50)
        
        # 의미론적 질의 확장 검색
        query = "흉부 증상이 있는 환자"
        # 여기서도 변수명 수정 (vector_store -> vs_builder)
        expanded_query = vs_builder.semantic_medical_query_expansion(query)
        semantic_results = vs_builder.search_similar_documents(expanded_query, vectorstore, k=3)
        
        print(f"\n의미론적 질의 확장 검색 쿼리: {query}")
        print(f"확장된 쿼리: {expanded_query}")
        print(f"검색 결과 ({len(semantic_results)}개):")
        for i, doc in enumerate(semantic_results, 1):
            print(f"\n결과 {i}:")
            print(f"환자 ID: {doc.metadata.get('patient_id', '알 수 없음')}")
            print(f"내용 미리보기: {doc.page_content[:200]}...")
            print("-" * 50)
        
        # 다중 인덱스 생성 (선택적)
        if input("다양한 벡터 인덱스를 생성하시겠습니까? (y/n): ").lower() == 'y':
            indices = vector_store.create_vector_indices()
            print(f"생성된 인덱스: {list(indices.keys())}")

if __name__ == "__main__":
    main()