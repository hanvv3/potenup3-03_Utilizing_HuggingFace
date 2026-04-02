# uv add streamlit-webrtc websocket-client opencv-python

import av
import base64
import json
import cv2
import websocket  # websocket-client (스레드 환경에서 동기 방식으로 사용)
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

WS_SERVER = "ws://localhost:8888"

st.title("웹캠 실시간 랜드마크")
st.caption("mediapipe 처리는 FastAPI 서버에서 수행됩니다.")


class LandmarkProcessor(VideoProcessorBase):
    def __init__(self):
        # 서버와 WebSocket 지속 연결
        self.ws = websocket.create_connection(f"{WS_SERVER}/ws/mediapipe")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 좌우 반전 (거울 모드)

        # STEP1. 프레임을 JPEG → base64로 인코딩
        _, buffer = cv2.imencode(".jpg", img)
        frame_b64 = base64.b64encode(buffer).decode()

        # STEP2. 서버로 전송 → 랜드마크 수신
        self.ws.send(json.dumps({"frame": frame_b64}))
        data = json.loads(self.ws.recv())
        landmarks = data["landmarks"]  # [[ {x, y, z}, ... ], ...]

        # STEP3. 랜드마크 점 그리기
        h, w = img.shape[:2]
        for hand in landmarks:
            for lm in hand:
                cx, cy = int(lm["x"] * w), int(lm["y"] * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="webcam",
    video_processor_factory=LandmarkProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
