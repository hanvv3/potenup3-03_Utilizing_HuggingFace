# uv add openai python-dotenv streamlit
# uv add streamlit==1.49.1
# .env 파일 만들어서 OPENAI_API_KEY 추가해두기
# 서버 실행: streamlit run main.py
import streamlit as st 

pages = [
    st.Page(
        page="pages/components.py",
        title="Basic",
        icon="😊",
        default=True
    )
]

nav = st.navigation(pages)
nav.run()