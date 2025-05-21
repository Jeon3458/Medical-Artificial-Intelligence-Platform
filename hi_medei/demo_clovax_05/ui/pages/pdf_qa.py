import mesop as me

from components.header import header
from components.pdf_upload import pdf_upload
from state.state import AppState


def pdf_qa_page(app_state: AppState):
    """PDF QA Page"""
    with me.box(
        style=me.Style(
            display='flex',
            flex_direction='column',
            height='100%',
        ),
    ):
        with me.box(
            style=me.Style(
                background=me.theme_var('background'),
                height='100%',
                margin=me.Margin(bottom=20),
            )
        ):
            with me.box(
                style=me.Style(
                    background=me.theme_var('background'),
                    padding=me.Padding(top=24, left=24, right=24, bottom=24),
                    display='flex',
                    flex_direction='column',
                    width='100%',
                )
            ):
                with header('PDF QA Agent', 'article'):
                    pass
                
                me.text(
                    "PDF QA Agent는 문서를 업로드하고 해당 문서에 대한 질문에 답변하는 에이전트입니다.",
                    type="body-1",
                    style=me.Style(margin=me.Margin(bottom=16))
                )
                
                pdf_upload()
                
                me.divider(style=me.Style(margin=me.Margin(top=16, bottom=16)))
                
                me.text(
                    "사용 방법:", 
                    type="headline-6",
                    style=me.Style(margin=me.Margin(bottom=8))
                )
                
                with me.box(style=me.Style(margin=me.Margin(left=16, bottom=16))):
                    me.text("1. PDF 파일 경로를 입력하고 '업로드 및 인제스트' 버튼을 클릭합니다.")
                    me.text("2. 'Remote Agents' 탭에서 'localhost:10000'을 추가합니다. (서버가 실행 중이어야 합니다)")
                    me.text("3. 'Conversation' 탭에서 PDF 문서에 대한 질문을 할 수 있습니다.")
                
                me.text(
                    "주의사항:", 
                    type="headline-6",
                    style=me.Style(margin=me.Margin(bottom=8))
                )
                
                with me.box(style=me.Style(margin=me.Margin(left=16))):
                    me.text("- PDF QA Agent 서버가 실행 중이어야 합니다: python -m agents.langgraph")
                    me.text("- OpenAI API 키가 환경변수 또는 .env 파일에 설정되어 있어야 합니다.") 