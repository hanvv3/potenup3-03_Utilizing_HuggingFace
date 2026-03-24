import streamlit as st
import requests

CHAT_URL = 'http://localhost:8080/chat'

st.title("챗봇 만들기")

# 저장소 만들기
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for chat in st.session_state['chat_history']:
    st.chat_message(chat['role']).markdown(chat['content'])

user_input = st.chat_input(
    placeholder='메세지를 입력하세요.'
)

data = {
    'message': user_input
}

response = requests.post(
    url=CHAT_URL,
    json=data
)
if response.status_code == 200:
    answer = response.json()['text']

if user_input:
    # 챗 표시
    st.chat_message('user').markdown(user_input)
    
    #answer = '안녕하세요.'
    st.chat_message('ai').markdown(answer)
    
    # 히스토리 저장
    st.session_state['chat_history'].extend(
        [
            {'role': 'user', 'content': user_input},
            {'role': 'ai', 'content': answer}
        ]
    )
