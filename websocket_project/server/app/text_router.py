import os
from fastapi import APIRouter, WebSocket
from fastapi.responses import StreamingResponse
from openai import OpenAI
from google import genai
import asyncio
from dotenv import load_dotenv
load_dotenv()

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Google Gemini 클라이언트 초기화
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 라우터
router = APIRouter()


@router.post("/chat")
async def post_chat(data: dict):
    # STEP1. 질문 받기
    question = data["question"]

    # STEP2. Gemini 응답 생성
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=question
    )
    result = response.text

    # STEP3. 결과 반환
    return {"result": result}


@router.post("/chat_stream")
async def post_chat_stream(data: dict):
    # STEP1. 질문 받기
    question = data.get("question", "")

    # STEP2-1. OpenAI 스트리밍 응답 생성
    # response = openai_client.chat.completions.create(
    #     model="gpt-5-nano",
    #     messages=[{"role": "user", "content": question}],
    #     stream=True,
    # )
    
    # STEP2-2. Google Gemini 스트리밍 응답 생성
    response = gemini_client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=question,
    )

    # STEP3-1. ChatGPT StreamingResponse로 토큰 단위 전송
    # def generator():
    #     for chunk in response:
    #         token = chunk.choices[0].delta.content
    #         if token is None:
    #             continue
    #         yield token
    
    # STEP3-2. Gemini StreamingResponse로 토큰 단위 전송
    async def generator():
        try:
            for chunk in response:
                # 텍스트가 존재하는지 확인 후 yield
                print(chunk)
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            # 스트리밍 중 에러 발생 처리
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(generator(), media_type="text/event-stream; charset=utf-8")


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        # STEP1. 데이터 받기
        json_data = await websocket.receive_json()
        question = json_data["question"]

        # STEP2. Gemini 스트리밍 응답 생성
        response = gemini_client.models.generate_content_stream(
            model="gemini-2.5-flash-lite",
            contents=question
        )

        # STEP3-1) chunk 단위로 전송
        for chunk in response:
            if chunk.text:
                print(chunk.text)  # 디버깅용 출력
                await websocket.send_json({"token": chunk.text})

        # STEP3-2) 한 글자씩 전송 (주석 해제 시 STEP3-1) 주석 처리 필요)
        # for chunk in response:
        #     if chunk.text:
        #         for char in chunk.text:
        #             await websocket.send_json({"token": char})

        # STEP4. 전송 완료 신호
        await websocket.send_json({"token": "[END]"})

    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    finally:
        await websocket.close()
        print("WebSocket 연결 종료")
