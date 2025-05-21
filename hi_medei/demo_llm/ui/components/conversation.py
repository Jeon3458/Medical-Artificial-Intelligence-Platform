import uuid
import os
import requests
import asyncio

import mesop as me

from common.types import Message, TextPart
from state.host_agent_service import (
    ListConversations,
    SendMessage,
    convert_message_to_state,
)
from state.state import AppState, SettingsState, StateMessage

from .chat_bubble import chat_bubble
from .form_render import form_sent, is_form, render_form


@me.stateclass
class PageState:
    """Local Page State"""

    conversation_id: str = ''
    message_content: str = ''
    pdf_upload_alert: str = ''
    show_pdf_alert: bool = False
    show_uploader: bool = False


def on_blur(e: me.InputBlurEvent):
    """Input handler"""
    state = me.state(PageState)
    state.message_content = e.value


async def send_message(message: str, message_id: str = ''):
    state = me.state(PageState)
    app_state = me.state(AppState)
    app_state.polling_interval = 1  # 입력 시 polling 재활성화!
    settings_state = me.state(SettingsState)
    c = next(
        (
            x
            for x in await ListConversations()
            if x.conversation_id == state.conversation_id
        ),
        None,
    )
    if not c:
        print('Conversation id ', state.conversation_id, ' not found')
    request = Message(
        id=message_id,
        role='user',
        parts=[TextPart(text=message)],
        metadata={
            'conversation_id': c.conversation_id if c else '',
            'conversation_name': c.name if c else '',
        },
    )
    # Add message to state until refresh replaces it.
    state_message = convert_message_to_state(request)
    if not app_state.messages:
        app_state.messages = []
    app_state.messages.append(state_message)
    conversation = next(
        filter(
            lambda x: x.conversation_id == c.conversation_id,
            app_state.conversations,
        ),
        None,
    )
    if conversation:
        conversation.message_ids.append(state_message.message_id)
    response = await SendMessage(request)
    # 답변이 모두 완료되면 polling을 꺼줌
    if not any(session_task.task.state != "COMPLETED" for session_task in app_state.task_list):
        app_state.polling_interval = 0


async def send_message_enter(e: me.InputEnterEvent):  # pylint: disable=unused-argument
    """Send message handler"""
    yield
    state = me.state(PageState)
    app_state = me.state(AppState)
    app_state.polling_interval = 1  # 엔터 입력 시 polling 재활성화!
    state.message_content = e.value
    message_id = str(uuid.uuid4())
    app_state.background_tasks[message_id] = ''
    yield
    await send_message(state.message_content, message_id)
    yield


async def send_message_button(e: me.ClickEvent):  # pylint: disable=unused-argument
    """Send message button handler"""
    yield
    state = me.state(PageState)
    app_state = me.state(AppState)
    app_state.polling_interval = 1  # 버튼 클릭 시 polling 재활성화!
    message_id = str(uuid.uuid4())
    app_state.background_tasks[message_id] = ''
    await send_message(state.message_content, message_id)
    yield


# PDF 업로드 핸들러 함수
def on_pdf_upload(e: me.UploadEvent):
    """PDF 파일 업로드 처리"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # PDF 파일인지 확인
    file_name = e.file.name
    if not file_name.lower().endswith('.pdf'):
        state.pdf_upload_alert = '오류: PDF 파일만 업로드 가능합니다'
        state.show_pdf_alert = True
        return
    
    try:
        # 현재 작업 디렉토리의 절대 경로 가져오기
        current_dir = os.path.abspath(os.path.dirname(__file__))
        # 프로젝트 루트 디렉토리 (A2A 디렉토리 위치)
        project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
        
        # 임시 디렉토리 생성 (여러 위치에 시도)
        temp_dirs = [
            os.path.join(project_root, "temp_uploads"),  # 프로젝트 루트
            os.path.join(project_root, "A2A/samples/python/temp_uploads"),  # 서버 실행 위치
            os.path.join(project_root, "A2A", "temp_uploads")  # A2A 디렉토리
        ]
        
        # 모든 가능한 위치에 디렉토리 생성 시도
        for temp_dir in temp_dirs:
            os.makedirs(temp_dir, exist_ok=True)
        
        # 각 위치에 파일 저장 시도
        saved = False
        file_content = e.file.read()
        for temp_dir in temp_dirs:
            try:
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                saved = True
                print(f"파일 저장 성공: {file_path}")
                break
            except Exception as ex:
                print(f"위치에 파일 저장 실패: {temp_dir}, 오류: {ex}")
                continue
                
        if not saved:
            raise Exception("모든 임시 디렉토리에 파일 저장 실패")
        
        # PDF 인제스트
        res = requests.post(
            'http://localhost:10000/ingest_pdf',
            json={'file_path': file_path},
            timeout=30
        )
        
        if res.ok:
            app_state.last_uploaded_pdf = file_name
            state.pdf_upload_alert = f'PDF 파일 "{file_name}" 업로드 및 인제스트 성공!'
            state.show_pdf_alert = True
            if file_name not in app_state.uploaded_pdfs:
                app_state.uploaded_pdfs.append(file_name)
                # 업로드 성공 메시지를 대화에 추가 - 비동기 처리
                # asyncio.create_task는 현재 컨텍스트에서 사용할 수 없으므로 대신 메시지만 설정
                state.message_content = f"PDF 파일 '{file_name}'을 업로드했습니다. 이제 파일에 대해 질문해 보세요."
                # 다음에 사용자가 UI와 상호작용할 때 메시지가 전송됨
        else:
            state.pdf_upload_alert = f'업로드 실패: {res.text}'
            state.show_pdf_alert = True
    except Exception as ex:
        state.pdf_upload_alert = f'업로드 실패: {ex}'
        state.show_pdf_alert = True


def close_alert(e: me.ClickEvent):
    """알림 닫기"""
    state = me.state(PageState)
    state.show_pdf_alert = False


def toggle_pdf_uploader(e: me.ClickEvent):
    """Toggle PDF uploader visibility"""
    state = me.state(PageState)
    state.show_uploader = not state.show_uploader


@me.component
def conversation():
    """Conversation component"""
    page_state = me.state(PageState)
    app_state = me.state(AppState)
    if 'conversation_id' in me.query_params:
        page_state.conversation_id = me.query_params['conversation_id']
        app_state.current_conversation_id = page_state.conversation_id
    
    with me.box(
        style=me.Style(
            display='flex',
            justify_content='space-between',
            flex_direction='column',
        )
    ):
        # PDF 업로드 알림 표시
        if page_state.show_pdf_alert:
            with me.box(
                style=me.Style(
                    padding=me.Padding.all(8),
                    margin=me.Margin(bottom=8),
                    background=me.theme_var('primary'),
                    color=me.theme_var('on-primary'),
                    border_radius=4,
                    display='flex',
                    justify_content='space-between',
                    align_items='center'
                )
            ):
                me.text(page_state.pdf_upload_alert)
                with me.content_button(
                    type='icon',
                    on_click=close_alert,
                    style=me.Style(color=me.theme_var('on-primary'))
                ):
                    me.icon(icon='close')
        
        for message in app_state.messages:
            if is_form(message):
                render_form(message, app_state)
            elif form_sent(message, app_state):
                chat_bubble(
                    StateMessage(
                        message_id=message.message_id,
                        role=message.role,
                        content=[('Form submitted', 'text/plain')],
                    ),
                    message.message_id,
                )
            else:
                chat_bubble(message, message.message_id)

        # PDF 업로더
        if page_state.show_uploader:
            me.uploader(
                label="PDF 업로드",
                on_upload=on_pdf_upload
            )

        with me.box(
            style=me.Style(
                display='flex',
                flex_direction='row',
                gap=5,
                align_items='center',
                min_width=500,
                width='100%',
                background=me.theme_var('surface-variant'),
                padding=me.Padding.all(8),
                border_radius=8,
            )
        ):
            me.input(
                label='How can I help you?',
                on_blur=on_blur,
                on_enter=send_message_enter,
                style=me.Style(min_width='75vw', flex_grow=1),
            )
            # PDF 업로드 버튼
            with me.content_button(
                type='flat',
                on_click=toggle_pdf_uploader,
                style=me.Style(
                    margin=me.Margin(right=8),
                    color=me.theme_var('primary')
                ),
            ):
                with me.tooltip(message="PDF 파일 업로드"):
                    me.icon(icon='description')
            # 메시지 전송 버튼
            with me.content_button(
                type='flat',
                on_click=send_message_button,
                style=me.Style(
                    color=me.theme_var('primary')
                ),
            ):
                me.icon(icon='send')
