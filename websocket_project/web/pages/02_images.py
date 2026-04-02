import asyncio
import base64
import json
import requests
import websockets
import streamlit as st

SERVER = "http://localhost:8888"
WS_SERVER = "ws://localhost:8888"

st.title("이미지 API 테스트")

# ────────────────────────────────────────────
# 섹션 1: POST /image — 단일 이미지 분석
# ────────────────────────────────────────────
st.subheader("1. POST /image — 단일 이미지 분석")

uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], key="up1")

if st.button("전송", key="btn1") and uploaded:
    # STEP1. 업로드된 이미지를 base64 문자열로 인코딩
    image_b64 = base64.b64encode(uploaded.read()).decode()
    # STEP2. 인코딩된 이미지를 JSON 바디로 POST 요청 전송
    res = requests.post(f"{SERVER}/image", json={"image": image_b64})
    # STEP3. 분석 결과 세션 저장
    st.session_state["out1"] = res.json()["result"]

with st.container(border=True):
    st.write(st.session_state.get("out1", ""))

st.divider()

# ────────────────────────────────────────────
# 섹션 2: WS /ws/mediapipe — 프레임 전송 & 랜드마크 수신
# ────────────────────────────────────────────
st.subheader("2. WS /ws/mediapipe — Mediapipe 손 랜드마크")

frame = st.camera_input("웹캠 캡처 버튼을 눌러 프레임 전송")

if frame:
    # STEP1. 캡처된 프레임을 base64 문자열로 인코딩
    frame_b64 = base64.b64encode(frame.getvalue()).decode()

    async def send():
        async with websockets.connect(f"{WS_SERVER}/ws/mediapipe") as ws:
            # STEP2. 인코딩된 프레임을 WebSocket으로 전송
            await ws.send(json.dumps({"frame": frame_b64}))
            # STEP3. 서버에서 Mediapipe 처리 후 랜드마크 결과 수신
            data = json.loads(await ws.recv())
            return data["landmarks"]

    loop = asyncio.new_event_loop()
    st.session_state["out2"] = loop.run_until_complete(send())
    loop.close()

with st.container(border=True):
    landmarks = st.session_state.get("out2")
    if not landmarks:
        st.write("손이 감지되지 않았습니다.")
    else:
        for i, hand in enumerate(landmarks):
            st.write(f"손 {i + 1}: 랜드마크 {len(hand)}개 감지")
            st.json(hand)
