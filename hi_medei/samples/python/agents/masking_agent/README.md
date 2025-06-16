# PHI 마스킹 에이전트 (Masking Agent)

Microsoft Presidio를 활용하여 한국어 의료 문서에서 개인정보(PHI)를 식별하고 마스킹하는 에이전트입니다. A2A 프로토콜을 통해 다른 에이전트들과 연동됩니다.

## 주요 기능

- **PHI 식별**

  - 환자 개인정보 식별
  - 의료 기록 식별
  - 한국어 특화 패턴 인식

- **데이터 마스킹**

  - 개인정보 자동 마스킹
  - 다양한 마스킹 옵션
  - 원본 데이터 보존

- **A2A 연동**
  - 표준 A2A 프로토콜 지원
  - 실시간 마스킹 처리
  - 다른 에이전트와 통합

## 설치 방법

1. 의존성 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

```bash
# Microsoft Presidio 설정 (필요시)
export PRESIDIO_API_KEY=your_api_key_here
```

## 실행 방법

```bash
# FastAPI 서버 시작
python start_masking_agent.py
```

## API 엔드포인트

- A2A 서버: `http://localhost:8000`
  - `/health`: 서버 상태 확인
  - `/a2a`: A2A 프로토콜 엔드포인트
  - `/mask`: 텍스트 마스킹 요청

## 마스킹 파이프라인

1. **텍스트 분석**

   - 한국어 텍스트 파싱
   - PHI 패턴 인식
   - 컨텍스트 분석

2. **개인정보 식별**

   - 이름, 주민번호 식별
   - 전화번호, 주소 식별
   - 의료 기록 식별

3. **마스킹 처리**

   - 식별된 정보 마스킹
   - 마스킹 레벨 조정
   - 원본 데이터 보존

4. **결과 반환**
   - 마스킹된 텍스트 생성
   - 마스킹 이력 기록
   - 메타데이터 포함

## 주의사항

- HIPAA 규정을 준수하여 PHI를 처리해야 합니다.
- 마스킹된 데이터는 원본을 복원할 수 없습니다.
- 개발/테스트 환경에서는 실제 환자 데이터를 사용하지 마세요.

## 라이선스

© 2024 Medical AI Platform Team. All rights reserved.
