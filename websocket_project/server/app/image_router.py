import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Mediapipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


@router.post("/image")
async def post_image(data: dict):
    """이미지 1장 수신 → 분석 결과 반환"""
    # STEP1. base64 → numpy 배열 변환
    image_bytes = base64.b64decode(data["image"])
    image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

    # STEP2. 처리하기
    result = "이미지를 수신했습니다."

    # STEP3. 결과 반환
    return {"result": result}


@router.websocket("/ws/mediapipe")
async def websocket_mediapipe(websocket: WebSocket):
    """웹캠 프레임 연속 수신 → mediapipe 처리 → 결과 반환
    요청(반복): {"frame": "base64_string"}
    응답(반복): {"landmarks": [...]}
    """
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        while True:
            # STEP1. 프레임 수신
            json_data = await websocket.receive_json()
            frame_b64 = json_data["frame"]

            # STEP2. base64 → numpy 배열 변환
            frame_bytes = base64.b64decode(frame_b64)
            frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

            # STEP3. Mediapipe 처리
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # STEP4. 랜드마크 추출
            landmarks = []
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    landmarks.append([
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand.landmark
                    ])

            # STEP5. 결과 전송
            await websocket.send_json({"landmarks": landmarks})

    except WebSocketDisconnect:
        print("클라이언트 연결 종료")
    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    finally:
        await websocket.close()
        print("WebSocket 연결 종료")
