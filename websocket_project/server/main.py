# 서버 실행: uvicorn main:app --port 8000 --reload

from fastapi import FastAPI
from app.text_router import router as text_router
from app.image_router import router as image_router
from app.audio_router import router as audio_router

app = FastAPI()

app.include_router(text_router)
app.include_router(image_router)
app.include_router(audio_router)


@app.get("/")
async def home():
    return {"hello": "world"}
