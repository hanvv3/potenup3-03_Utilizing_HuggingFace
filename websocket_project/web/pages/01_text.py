import time
import asyncio
import json
import requests
import websockets
import streamlit as st

SERVER = "http://localhost:8888"
WS_SERVER = "ws://localhost:8888"

st.title("텍스트 API 테스트")

# ────────────────────────────────────────────
# 섹션 1: POST /chat — Gemini 단일 응답
# ────────────────────────────────────────────
st.subheader("1. POST /chat — Gemini 단일 응답")

q1 = st.text_input("질문", key="q1")

if st.button("전송", key="btn1"):
    # STEP1. 질문을 JSON 바디로 POST 요청 전송
    res = requests.post(f"{SERVER}/chat", json={"question": q1})
    # STEP2. 응답에서 결과 추출 후 세션 저장
    st.session_state["out1"] = res.json()["result"]

# STEP3. 결과 출력 (세션에서 불러오기)
with st.container(border=True):
    st.write(st.session_state.get("out1", ""))

st.divider()

# ────────────────────────────────────────────
# 섹션 2: POST /chat_stream — OpenAI 스트리밍
# ────────────────────────────────────────────
st.subheader("2. POST /chat_stream — OpenAI/Gemini 스트리밍")

q2 = st.text_input("질문", key="q2")
placeholder = st.empty()

if st.button("전송", key="btn2"):
    # STEP1. stream=True 로 POST 요청 — 서버가 토큰 단위로 응답을 흘려줌
    res = requests.post(f"{SERVER}/chat_stream", json={"question": q2}, stream=True)
    
    # 청크를 넘겨주기 위한 제너레이터 함수
    def stream_chunks():
        for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                for char in chunk:
                    yield char
                    time.sleep(0.01)

    # STEP2. st.write_stream으로 실시간 화면 업데이트 (타이핑 효과 자동 적용)
    with placeholder.container(border=True):
        # st.write_stream은 제너레이터를 받아 화면에 그리고, 최종 완성된 문자열을 반환합니다.
        text = st.write_stream(stream_chunks())
        
    # STEP3. 최종 전체 텍스트를 세션 저장
    st.session_state["out2"] = text
else:
    # 버튼 클릭 전에는 세션에서 결과 불러와서 출력
    with placeholder.container(border=True):
        st.write(st.session_state.get("out2", ""))

st.divider()

# ────────────────────────────────────────────
# 섹션 3: WS /ws/chat — WebSocket
# ────────────────────────────────────────────
st.subheader("3. WS /ws/chat — WebSocket")

q3 = st.text_input("질문", key="q3")

if st.button("전송", key="btn3"):
    async def send():
        async with websockets.connect(f"{WS_SERVER}/ws/chat") as ws:
            # STEP1. 질문을 JSON으로 WebSocket 전송
            await ws.send(json.dumps({"question": q3}))
            text = ""
            while True:
                # STEP2. 토큰 수신 — [END] 신호를 받으면 루프 종료
                data = json.loads(await ws.recv())
                if data["token"] == "[END]":
                    break
                text += data["token"]
            return text

    # STEP3. 이벤트 루프 실행 & 결과 세션 저장
    loop = asyncio.new_event_loop()
    st.session_state["out3"] = loop.run_until_complete(send())
    loop.close()

# STEP4. 결과 출력 (세션에서 불러오기)
with st.container(border=True):
    st.write(st.session_state.get("out3", ""))
