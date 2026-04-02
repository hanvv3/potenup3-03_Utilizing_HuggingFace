import asyncio
import base64
import json
import requests
import websockets
import streamlit as st

SERVER = "http://localhost:8888"
WS_SERVER = "ws://localhost:8888"

st.title("오디오 API 테스트")

# ────────────────────────────────────────────
# 섹션 1: POST /stt — Whisper STT
# ────────────────────────────────────────────
st.subheader("1. POST /stt — Whisper STT")

audio1 = st.audio_input("녹음", key="up1")

if st.button("전송", key="btn1") and audio1:
    # STEP1. 오디오 파일을 multipart/form-data 로 POST 전송
    res = requests.post(f"{SERVER}/stt", files={"audio": audio1})
    # STEP2. STT 변환 결과 세션 저장
    st.session_state["out1"] = res.json()["result"]

with st.container(border=True):
    st.write(st.session_state.get("out1", ""))

st.divider()

# ────────────────────────────────────────────
# 섹션 2: WS /ws/stt — WebSocket STT (청크 전송)
# ────────────────────────────────────────────
st.subheader("2. WS /ws/stt — WebSocket STT")

audio2 = st.audio_input("녹음", key="up2")

if st.button("전송", key="btn2") and audio2:
    audio_bytes = audio2.read()

    async def send_stt():
        async with websockets.connect(f"{WS_SERVER}/ws/stt") as ws:
            # STEP1. 오디오를 4096 bytes 청크로 분할하여 순차 전송
            for i in range(0, len(audio_bytes), 4096):
                chunk = audio_bytes[i:i + 4096]
                await ws.send(json.dumps({"chunk": base64.b64encode(chunk).decode()}))
                await ws.recv()  # {"status": "received"} 수신 — 흐름 제어용
            # STEP2. 전송 완료 신호를 보내 서버가 STT 처리를 시작하도록 알림
            await ws.send(json.dumps({"done": True}))
            # STEP3. 최종 STT 결과 수신
            data = json.loads(await ws.recv())
            return data["result"]

    loop = asyncio.new_event_loop()
    st.session_state["out2"] = loop.run_until_complete(send_stt())
    loop.close()

with st.container(border=True):
    st.write(st.session_state.get("out2", ""))

st.divider()

# ────────────────────────────────────────────
# 섹션 3: POST /tts — OpenAI TTS StreamingResponse
# ────────────────────────────────────────────
st.subheader("3. POST /tts — OpenAI TTS")

text3 = st.text_input("텍스트 입력", key="t3")

if st.button("전송", key="btn3") and text3:
    # STEP1. 텍스트를 POST 요청으로 전송 — 서버가 StreamingResponse 로 오디오 bytes 반환
    res = requests.post(f"{SERVER}/tts", json={"text": text3})
    # STEP2. 오디오 bytes 세션 저장
    st.session_state["out3"] = res.content

# STEP3. 저장된 오디오 재생
if st.session_state.get("out3"):
    st.audio(st.session_state["out3"], format="audio/mp3")

st.divider()

# ────────────────────────────────────────────
# 섹션 4: WS /ws/tts — WebSocket TTS
# ────────────────────────────────────────────
st.subheader("4. WS /ws/tts — WebSocket TTS")

text4 = st.text_input("텍스트 입력", key="t4")

if st.button("전송", key="btn4") and text4:
    async def send_tts():
        async with websockets.connect(f"{WS_SERVER}/ws/tts") as ws:
            # STEP1. 텍스트를 WebSocket으로 전송
            await ws.send(json.dumps({"text": text4}))
            audio_chunks = []
            while True:
                # STEP2. 오디오 청크 수신 — done 신호를 받으면 루프 종료
                data = json.loads(await ws.recv())
                if data.get("done"):
                    break
                audio_chunks.append(base64.b64decode(data["chunk"]))
            # STEP3. 수신된 청크를 하나로 합쳐 완성된 오디오 반환
            return b"".join(audio_chunks)

    loop = asyncio.new_event_loop()
    st.session_state["out4"] = loop.run_until_complete(send_tts())
    loop.close()

# STEP4. 조립된 오디오 재생
if st.session_state.get("out4"):
    st.audio(st.session_state["out4"], format="audio/mp3")
