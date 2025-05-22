try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    raise ImportError("Presidio 라이브러리가 설치되어 있지 않습니다. requirements.txt 참고해서 설치하세요.")

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def run(input: dict) -> dict:
    text = input.get("text", "")
    results = analyzer.analyze(text=text, language='en')
    redacted_text = anonymizer.anonymize(text=text, analyzer_results=results).text
    return {"masked_text": redacted_text}
