#!/usr/bin/env python3
"""
Medical Image Agent A2A Server
OpenAI Vision + HyperCLOVAX 파이프라인을 A2A 프로토콜로 제공
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# 올바른 Python path 설정
# 현재: /hi_medei/samples/python/agents/medical_image_agent
# common: /hi_medei/samples/python/common
python_root = Path(__file__).resolve().parents[2]  # /hi_medei/samples/python
sys.path.insert(0, str(python_root))

# .env 파일 로드
from dotenv import load_dotenv
env_path = python_root / ".env"
load_dotenv(env_path)

from common.server.server import A2AServer
from common.types import AgentCard, AgentProvider, AgentCapabilities, AgentSkill

from task_manager import MedicalImageTaskManager


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def create_agent_card() -> AgentCard:
    """의료 영상 분석 에이전트 카드 생성"""
    return AgentCard(
        name="Medical Image Analysis Agent",
        description="OpenAI Vision API를 사용한 의료 영상 분석 에이전트",
        url="http://localhost:10002",
        provider=AgentProvider(
            organization="Medical AI Lab"
        ),
        version="2.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text", "file"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="medical_image_analysis",
                name="의료 영상 분석",
                description="X-Ray, CT, MRI, 초음파 등 의료 영상을 분석하여 소견을 제공합니다",
                tags=["medical", "radiology", "imaging", "AI", "OpenAI", "HyperCLOVAX"],
                examples=[
                    "흉부 X-Ray에서 폐렴 징후 분석",
                    "복부 CT에서 종양 의심 부위 식별",
                    "뇌 MRI에서 이상 소견 검출"
                ],
                inputModes=["text", "file"],
                outputModes=["text"]
            )
        ]
    )


def main():
    """메인 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🏥 Medical Image Analysis Agent 시작")
    
    # 환경 변수 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")
        return
    
    logger.info("✅ OpenAI API Key 확인됨")
    
    try:
        # Task Manager 초기화
        task_manager = MedicalImageTaskManager()
        
        # Agent Card 생성
        agent_card = create_agent_card()
        
        # A2A 서버 생성 및 시작
        server = A2AServer(
            host="localhost",
            port=10002,
            endpoint="/",
            agent_card=agent_card,
            task_manager=task_manager
        )
        
        logger.info("🚀 A2A 서버를 localhost:10002에서 시작합니다...")
        server.start()
        
    except KeyboardInterrupt:
        logger.info("⛔ 서버가 중단되었습니다")
    except Exception as e:
        logger.error(f"❌ 서버 실행 오류: {e}")
        raise


if __name__ == "__main__":
    main() 