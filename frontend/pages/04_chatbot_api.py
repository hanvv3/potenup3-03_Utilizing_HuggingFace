import streamlit as st
import requests

CHAT_URL = 'http://localhost:8080/chat'

st.title("자기가 사용자라고 착각하는 챗봇 만들기")

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

try:
    response = requests.post(
        url=CHAT_URL,
        json=data
    )
    if response.status_code == 200:
        answer = response.json()['text']
except:
    answer = 'LLM에 접근할 수 없는 상태입니다.'

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
