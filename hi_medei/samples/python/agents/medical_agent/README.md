# 의료 에이전트 (Medical Agent)

의료 환경에서 환자 데이터 관리, SOAP 노트 생성, 의료 검색 등을 수행하는 AI 에이전트입니다.

## 주요 기능

- **환자 데이터 관리**

  - 환자 정보 검색 및 조회
  - 환자 기록 관리
  - SOAP 노트 생성 및 관리

- **의료 검색**

  - PubMed 기반 의학 문헌 검색
  - 임상 가이드라인 검색
  - 의학 용어 및 개념 검색

- **MCP 서버 연동**
  - PubMed 서버 (포트 8080)
  - 메모리 서버 (포트 8081)
  - 벡터 검색 기능

## 설치 방법

1. 의존성 설치:

```bash
# 기본 기능만 사용시
pip install -r requirements_simplified.txt

# 모든 기능 사용시
pip install -r requirements_complete.txt

# MCP 서버 사용시
pip install -r requirements_mcp.txt
```

2. 환경 변수 설정:

```bash
cp env_example.txt .env
# .env 파일을 편집하여 필요한 API 키 설정
```

## 실행 방법

### 통합 서버 시작 (MCP 서버 포함)

```bash
python start_all.py
```

### 개별 서버 시작

```bash
# A2A 서버 시작
python __main__.py

# MCP 서버 시작 (별도 터미널)
python mcp_servers/start_mcp.py
```

## API 엔드포인트

- A2A 서버: `http://localhost:10001`

  - `/health`: 서버 상태 확인
  - `/a2a`: A2A 프로토콜 엔드포인트

- MCP 서버:
  - PubMed 서버: `http://localhost:8080`
  - 메모리 서버: `http://localhost:8081`

## 테스트

```bash
# 기본 테스트
python test_agent_simple.py

# MCP 연동 테스트
python test_agent_mcp.py
python test_mcp_connection.py
python test_mcp_integration.py

# 벡터 검색 테스트
python test_vector.py
```

## 디버그 모드

```bash
# 에이전트 호출 디버그
python debug_invoke.py

# 검색 기능 디버그
python debug_search.py

# 쿼리 분석 디버그
python debug_query.py
```

## 주의사항

- HIPAA 준수를 위해 환자 데이터는 반드시 암호화되어야 합니다.
- 개발/테스트 환경에서는 실제 환자 데이터를 사용하지 마세요.
- API 키는 안전하게 관리해야 합니다.

## 라이선스

© 2024 Medical AI Platform Team. All rights reserved.
