# 의료 영상 분석 에이전트 (Medical Image Agent)

의료 영상(X-Ray, CT, MRI 등)을 AI로 분석하고 진단을 지원하는 에이전트입니다. A2A 프로토콜과 MCP를 통해 PACS 시스템과 연동됩니다.

## 주요 기능

- **의료 영상 분석**

  - X-Ray 이미지 분석
  - CT 스캔 분석
  - MRI 이미지 분석
  - 이상 소견 감지 및 분류

- **진단 지원**

  - 영상 기반 진단 제안
  - 병변 위치 표시
  - 진단 확률 제공
  - 관련 의학 문헌 추천

- **PACS 연동**
  - DICOM 파일 처리
  - PACS 시스템 연동
  - 영상 메타데이터 관리

## 설치 방법

1. 의존성 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY=your_api_key_here

# Google API 키 설정 (선택)
export GOOGLE_API_KEY=your_api_key_here
```

## 실행 방법

```bash
# A2A 서버 시작
python __main__.py
```

## API 엔드포인트

- A2A 서버: `http://localhost:10002`
  - `/health`: 서버 상태 확인
  - `/a2a`: A2A 프로토콜 엔드포인트
  - `/analyze`: 영상 분석 요청
  - `/status`: 분석 상태 확인

## 영상 분석 파이프라인

1. **영상 수신**

   - DICOM 파일 또는 이미지 파일 수신
   - 파일 형식 검증
   - 메타데이터 추출

2. **전처리**

   - 이미지 정규화
   - 노이즈 제거
   - 해상도 조정

3. **분석**

   - AI 모델을 통한 영상 분석
   - 이상 소견 감지
   - 진단 확률 계산

4. **결과 생성**
   - 분석 보고서 생성
   - 시각화 결과 생성
   - 관련 의학 문헌 추천

## 주의사항

- 의료 영상 데이터는 HIPAA 규정을 준수하여 처리해야 합니다.
- AI 분석 결과는 참고용이며, 최종 진단은 의료 전문가가 내려야 합니다.
- 개발/테스트 환경에서는 실제 환자 영상을 사용하지 마세요.

## 라이선스

© 2024 Medical AI Platform Team. All rights reserved.
