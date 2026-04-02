import base64
import os
import tempfile

from fastapi import APIRouter, File, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()


@router.post("/stt")
async def post_stt(audio: UploadFile = File(...)):
    """오디오 파일 → 텍스트 (Whisper)"""
    # STEP1. 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio.read())

    try:
        # STEP2. Whisper STT 실행
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        # STEP3. 결과 반환
        return {"result": result.text}
    finally:
        os.remove(tmp_path)


@router.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    """오디오 청크 수신 → 텍스트 반환 (Whisper)
    요청(반복): {"chunk": "base64_string"}
    요청(완료): {"done": true}
    응답(수신 확인): {"status": "received"}
    응답(최종 결과): {"result": "텍스트"}
    """
    await websocket.accept()
    print("WebSocket 연결됨")

    audio_chunks = []
    tmp_path = None

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("done"):
                # STEP2. 청크 합쳐서 임시 파일 저장
                audio_bytes = b"".join(audio_chunks)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp.write(audio_bytes)

                # STEP3. Whisper STT 실행 후 결과 전송
                with open(tmp_path, "rb") as f:
                    result = openai_client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f
                    )
                await websocket.send_json({"result": result.text})
                break
            else:
                # STEP1. 청크 수신 및 누적
                audio_chunks.append(base64.b64decode(data["chunk"]))
                await websocket.send_json({"status": "received"})

    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        await websocket.close()
        print("WebSocket 연결 종료")


@router.post("/tts")
async def post_tts(data: dict):
    """텍스트 → 오디오 스트리밍 (OpenAI TTS)"""
    # STEP1. 텍스트 받기
    text = data["text"]

    # STEP2. TTS 스트리밍 응답 생성
    def generator():
        with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="mp3",
        ) as response:
            # STEP3. 오디오 청크 단위로 전송
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

    return StreamingResponse(generator(), media_type="audio/mpeg")


@router.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """텍스트 수신 → 오디오 청크 전송 (OpenAI TTS)
    요청: {"text": "텍스트"}
    응답(반복): {"chunk": "base64_string"}
    응답(완료): {"done": true}
    """
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        # STEP1. 텍스트 받기
        json_data = await websocket.receive_json()
        text = json_data["text"]

        # STEP2. TTS 스트리밍 생성 후 청크 전송
        with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="mp3",
        ) as response:
            # STEP3. base64로 인코딩하여 청크 단위 전송
            for chunk in response.iter_bytes(chunk_size=4096):
                await websocket.send_json({"chunk": base64.b64encode(chunk).decode()})

        # STEP3. 전송 완료 신호
        await websocket.send_json({"done": True})

    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    finally:
        await websocket.close()
        print("WebSocket 연결 종료")
