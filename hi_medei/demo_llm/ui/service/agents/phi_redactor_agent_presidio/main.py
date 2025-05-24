# main.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Analyzer / Anonymizer 세팅 (NLP 엔진 없이)
analyzer = AnalyzerEngine(
    supported_languages=["en", "ko"],
    nlp_engine=None  # NLP 엔진 명시적으로 비활성화
)
anonymizer = AnonymizerEngine()

# 1. 한국 전화번호 패턴
kr_phone_patterns = [
    Pattern(name="KR_PHONE_DASH", regex=r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}", score=0.9),
    Pattern(name="KR_PHONE_NO_DASH", regex=r"01[016789]\d{7,8}", score=0.9),
]

kr_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=kr_phone_patterns,
    supported_language="ko"
)

kr_phone_recognizer_en = PatternRecognizer(
    supported_entity="PHONE_NUMBER", 
    patterns=kr_phone_patterns,
    supported_language="en"
)

# 2. 생일 패턴들 (한글 + 영어)
birthday_patterns_ko = [
    Pattern(name="DATE_KO_YMD", regex=r"\b\d{4}년\s?\d{1,2}월\s?\d{1,2}일\b", score=0.95),
    Pattern(name="DATE_KO_MD", regex=r"\b\d{1,2}월\s?\d{1,2}일\b", score=0.8),
    Pattern(name="DATE_KO_BIRTH", regex=r"\b(19|20)\d{2}년생\b", score=0.9),
    Pattern(name="DATE_YYMMDD", regex=r"\b\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b", score=0.7),
    Pattern(name="DATE_YYYYMMDD", regex=r"\b(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b", score=0.8),
]

birthday_patterns_en = [
    Pattern(name="DATE_YYYY_MM_DD", regex=r"\b\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2}\b", score=0.9),
    Pattern(name="DATE_MM_DD_YYYY", regex=r"\b\d{1,2}[-/\s]\d{1,2}[-/\s]\d{4}\b", score=0.9),
    Pattern(name="DATE_MONTH_DD_YYYY", regex=r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", score=0.9),
]

birthday_recognizer_ko = PatternRecognizer(
    supported_entity="DATE_TIME",
    patterns=birthday_patterns_ko,
    supported_language="ko"
)

birthday_recognizer_en = PatternRecognizer(
    supported_entity="DATE_TIME",
    patterns=birthday_patterns_en,
    supported_language="en"
)

# 3. 한글 이름 패턴
name_patterns_ko = [
    # 한글 이름 (성+이름)
    Pattern(name="KR_NAME_FULL", regex=r"\b[가-힣]{1,2}[가-힣]{1,3}\b", score=0.7),
    # 일반적인 한국 성씨 + 이름
    Pattern(name="KR_NAME_COMMON", regex=r"\b(김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍|고|문|양|손|배|조|백|허|유|남|심|노|정|하|곽|성|차|주|우|구|신|민|유|진|엄|원|천|방|공|현|함|변|염|양|천|석|설|길|탁|남궁|사공)[가-힣]{1,3}\b", score=0.85),
    # 직함/호칭 + 이름
    Pattern(name="KR_NAME_TITLE", regex=r"\b(님|씨|선생|교수|박사|의사|변호사|대표|사장|부장|과장|대리|주임|팀장)\s*[가-힣]{2,4}\b", score=0.8),
]

# 영어 이름 패턴 (한국식 영어 이름)
name_patterns_en = [
    # 한국식 영어 이름 (성이 뒤에 오는 경우)
    Pattern(name="KR_EN_NAME_LAST", regex=r"\b[A-Z][a-z]+(seong|sung|jin|min|hyun|hyeon|woo|soo|hee|mi|jung|young|kyung|tae|jae|ho|sun|moon)\s+(Kim|Lee|Park|Choi|Jung|Kang|Yoon|Jeon|Shin|Oh|Kwon|Hwang|Ahn|Song|Hong|Yoo|Han|Seo|Ko|Moon|Lim|Cho|Bae|Ryu|Yang|Nam|Baek)\b", score=0.95),
]

name_recognizer_ko = PatternRecognizer(
    supported_entity="PERSON",
    patterns=name_patterns_ko,
    supported_language="ko"
)

name_recognizer_en = PatternRecognizer(
    supported_entity="PERSON",
    patterns=name_patterns_en,
    supported_language="en"
)

# Recognizer 등록
# 한국어 recognizers
analyzer.registry.add_recognizer(kr_phone_recognizer)
analyzer.registry.add_recognizer(birthday_recognizer_ko)
analyzer.registry.add_recognizer(name_recognizer_ko)

# 영어 recognizers
analyzer.registry.add_recognizer(kr_phone_recognizer_en)
analyzer.registry.add_recognizer(birthday_recognizer_en)
analyzer.registry.add_recognizer(name_recognizer_en)

# 마스킹 함수
def run(input: dict) -> dict:
    text = input.get("text", "")
    
    # 한글이 있는지 확인
    has_korean = any('\uac00' <= char <= '\ud7af' for char in text)
    language = "ko" if has_korean else "en"
    
    # 텍스트 분석
    results = analyzer.analyze(
        text=text,
        language=language,
        return_decision_process=False,
        nlp_artifacts=None  # NLP 아티팩트 명시적으로 None 설정
    )
    
    # 결과 정렬 (위치 순서대로)
    results = sorted(results, key=lambda x: x.start)
    
    # 디버깅: 어떤 항목이 인식되었는지 출력
    print(f"\n=== 인식된 항목들 (언어: {language}) ===")
    for result in results:
        print(f"Entity: {result.entity_type}, Text: '{text[result.start:result.end]}', Score: {result.score}")
    
    # 마스킹 연산자 설정
    operators = {
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "***"}),
        "DATE_TIME": OperatorConfig("replace", {"new_value": "***"}),
        "PERSON": OperatorConfig("replace", {"new_value": "***"})
    }
    
    # 텍스트 익명화
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )
    
    return {"masked_text": anonymized_result.text}