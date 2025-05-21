import mesop as me
import requests
import os
import base64
from state.state import AppState

# 파일 업로드 핸들러
def on_file_upload(e: me.UploadEvent):
    state = me.state(AppState)
    
    # 파일 이름에서 확장자 확인
    file_name = e.file.name
    
    # PDF 파일인지 확인 (확장자로)
    if not file_name.lower().endswith('.pdf'):
        state.pdf_upload_message = '오류: PDF 파일만 업로드 가능합니다'
        return
    
    # 임시 파일로 저장 (절대 경로 사용)
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
            except Exception as e:
                print(f"위치에 파일 저장 실패: {temp_dir}, 오류: {e}")
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
            state.last_uploaded_pdf = file_name
            state.pdf_upload_message = '업로드 및 인제스트 성공!'
            if file_name not in state.uploaded_pdfs:
                state.uploaded_pdfs.append(file_name)
        else:
            state.pdf_upload_message = f'업로드 실패: {res.text}'
    except Exception as e:
        state.pdf_upload_message = f'업로드 실패: {e}'

@me.component
def pdf_upload():
    state = me.state(AppState)
    
    # 파일 업로드(드래그앤드롭) 섹션
    me.text("PDF 파일 업로드", type="subtitle-1", style=me.Style(margin=me.Margin(bottom=8)))
    
    with me.box(style=me.Style(margin=me.Margin(bottom=16))):
        me.uploader(
            label="PDF 파일을 드래그하거나 클릭하여 업로드하세요",
            on_upload=on_file_upload,
            style=me.Style(width="100%")
        )
    
    # 메시지 표시
    if state.pdf_upload_message:
        me.text(state.pdf_upload_message, style=me.Style(color='green' if '성공' in state.pdf_upload_message else 'red'))
    
    # 업로드된 파일 목록
    if state.uploaded_pdfs:
        with me.box():
            me.text('업로드된 PDF 목록:', type="subtitle-1")
            for pdf in state.uploaded_pdfs:
                with me.box(style=me.Style(display='flex')):
                    me.icon(icon="description")
                    me.text(pdf)
    elif state.last_uploaded_pdf:
        me.text(f'최근 업로드: {state.last_uploaded_pdf}') 