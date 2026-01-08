from .host_agent import HostAgent


# PDF QA 에이전트 자동 등록 제거 - A2A 협업으로 처리
root_agent = HostAgent([]).create_agent()
