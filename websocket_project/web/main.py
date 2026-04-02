# 실행: streamlit run main.py

import streamlit as st

# STEP1. 페이지 목록 정의 — 각 파일을 사이드바 메뉴 항목으로 등록
pages = [
    st.Page("pages/01_text.py", title="텍스트"),
    st.Page("pages/02_images.py", title="이미지"),
    st.Page("pages/03_audio.py", title="오디오"),
    st.Page("pages/04_webcam.py", title="웹캠"),
    st.Page("pages/05_ai_speaker.py", title="AI 스피커"),
]

# STEP2. 네비게이션 생성 & 선택된 페이지 실행
pg = st.navigation(pages)
pg.run()
