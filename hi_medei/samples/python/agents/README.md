# Hi Medei AI Agents

이 디렉토리는 Hi Medei 플랫폼의 다양한 AI 에이전트들을 포함하고 있습니다. 각 에이전트는 독립적인 A2A 서버로 실행되며, 의료 환경에서의 다양한 작업을 수행할 수 있습니다.

## 구현된 에이전트

- [**의료 에이전트 (Medical Agent)**](/samples/python/agents/medical_agent/README.md)  
  환자 데이터 관리, SOAP 노트 생성, 의료 검색 등을 수행하는 의료 AI 에이전트입니다. MCP 서버와 연동되어 PubMed 검색 및 메모리 관리 기능을 제공합니다.

- [**의료 영상 분석 에이전트 (Medical Image Agent)**](/samples/python/agents/medical_image_agent/README.md)  
  X-Ray, CT, MRI 등의 의료 영상을 AI로 분석하고 진단을 지원하는 에이전트입니다. A2A 프로토콜과 MCP를 통해 PACS 시스템과 연동됩니다.

- [**PDF QA 에이전트 (LangGraph)**](/samples/python/agents/langgraph/README.md)  
  의료 문서와 PDF 파일을 분석하고 질의응답을 수행하는 에이전트입니다. LangGraph 프레임워크를 사용하여 구현되었으며, 벡터 데이터베이스를 활용한 효율적인 문서 검색을 제공합니다.

- [**PHI 마스킹 에이전트 (Masking Agent)**](/samples/python/agents/masking_agent/README.md)  
  Microsoft Presidio를 활용하여 한국어 의료 문서에서 개인정보(PHI)를 식별하고 마스킹하는 에이전트입니다. A2A 프로토콜을 통해 다른 에이전트들과 연동됩니다.

## 공통 기능

- 모든 에이전트는 A2A 프로토콜을 준수하여 구현되었습니다.
- 각 에이전트는 독립적인 포트에서 실행됩니다 (기본 포트는 각 README에서 확인 가능).
- 에이전트 간 통신은 A2A 프로토콜을 통해 이루어집니다.
- MCP 서버와의 연동을 통해 확장된 기능을 제공합니다.

## 사용 방법

1. 각 에이전트의 README.md를 참조하여 필요한 의존성을 설치합니다.
2. 환경 변수를 설정합니다 (API 키 등).
3. 각 에이전트의 시작 스크립트를 실행합니다.
4. A2A 클라이언트를 통해 에이전트와 상호작용합니다.

## 개발 가이드

- 새로운 에이전트 개발 시 A2A 프로토콜 스펙을 준수해야 합니다.
- 테스트는 각 에이전트 디렉토리의 테스트 파일을 통해 수행할 수 있습니다.
- 디버그 모드 실행 방법은 각 에이전트의 README.md를 참조하세요.

## 라이선스

© 2024 Medical AI Platform Team. All rights reserved.
