# uv add openai python-dotenv streamlit
# uv add streamlit==1.55.0
# .env 파일 만들어서 OPENAI_API_KEY 추가해두기
# 서버 실행: streamlit run main.py
import streamlit as st 

pages = [
    st.Page(
        page="pages/components.py",
        title="Basic",
        icon="😊",
        default=True
    ),
    st.Page(
        page="pages/whatis0.py",
        title="What is 'Object Detection'?",
        icon="😊"
    ),
    st.Page(
        page="pages/01_object_detection.py",
        title="Object Detection",
        icon="😊"
    ),
    st.Page(
        page="pages/whatis1.py",
        title="What is 'Segmentation'?",
        icon="😊"
    ),
    st.Page(
        page="pages/02_segmentation.py",
        title="Segmentation",
        icon="😊"
    ),
]

nav = st.navigation(pages)
nav.run()