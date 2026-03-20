from fastapi import FastAPI, UploadFile, File, Form # Form 추가!
from contextlib import asynccontextmanager
import shutil
import os
import io
import numpy as np
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor

# -----------------------
# 설정
# -----------------------
UPLOAD_DIR = "upload_img"
os.makedirs(UPLOAD_DIR, exist_ok=True) # 실행 위치(현재 폴더)에 생성됨

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("============= Model loading initiated =============")
    def load_clip_model():
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        return model, processor
    
    clip_model, processor = load_clip_model()
    app.state.clip_model = clip_model.to(device)
    app.state.processor = processor
    
    overrides = dict(
        task='segment',
        mode='predict',
        model='../models/sam3.pt',
        conf=0.25
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    app.state.predictor = predictor
    print("================== Model Ready ====================")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    # 💡 수정: 경로를 현재 폴더 내 UPLOAD_DIR로 통일
    file_name = f"./{UPLOAD_DIR}/{now}-{file.filename}"

    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": now
    }

@app.post("/detect_image")
async def predict_sam( # 함수 이름도 직관적으로 yolo -> sam으로 변경 (선택사항)
    # 💡 수정: dict 대신 str 타입으로 Form 데이터를 받도록 변경
    text: str = Form(..., description="분할할 객체 프롬프트 (예: 'car')"), 
    file: UploadFile = File(...)
):
    # 1. img 읽기
    file_bytes = await file.read()
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    img_bgr = np.array(img)[:, :, ::-1] 
    app.state.predictor.set_image(img_bgr)
    
    # 2. 예측하기 (Form으로 받은 text(문자열)를 넘겨줌)
    results = app.state.predictor(text=[text])    
    result = results[0]

    # 3. 데이터 만들기 (FastAPI의 /detect_image 내부)
    detections = []
    names = result.names 
    
    # 박스와 마스크가 모두 있을 때 처리
    if result.boxes is not None and result.masks is not None:
        for i in range(len(result.boxes)):
            box = result.boxes.xyxy[i].cpu().numpy().tolist() # [x1, y1, x2, y2]
            conf = result.boxes.conf[i].item()
            label_idx = int(result.boxes.cls[i].item())
            
            # 💡 핵심: 마스크 폴리곤(다각형) 좌표 추출
            polygon = result.masks.xy[i].tolist() 
            
            detections.append(
                {
                    "box": box,
                    "confidence": conf,
                    "label": names[label_idx] if names else text,
                    "polygon": polygon # 마스크 좌표 추가!
                }
            )

    # 4. 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./{UPLOAD_DIR}/{now}-{file.filename}"

    # 5. 파일 저장
    await file.seek(0)
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 성공적으로 분석하고 저장했습니다.",
        "filename": file_name,
        "time": now,
        "object_detection": detections
    }

@app.post("/similarity")
async def predict_clip( # 함수 이름도 직관적으로 yolo -> sam으로 변경 (선택사항)
    # 💡 수정: dict 대신 str 타입으로 Form 데이터를 받도록 변경
    text: str = Form(..., description="분할할 객체 프롬프트 (예: 'car')"), 
    file: UploadFile = File(...)
):
    # 1. img 읽기
    file_bytes = await file.read()
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    classes = [label.strip() for label in text.split(',') if label.strip()]
    
    inputs = app.state.processor(
        text=classes,
        images=img,
        return_tensors='pt',
        padding=True
    )
    
    # tensor를 모델이 있는 cpu/gpu장치로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. 예측하기 (Form으로 받은 text(문자열)를 넘겨줌)
    with torch.inference_mode():
        outputs = app.state.clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    # 3. 데이터 만들기 (FastAPI의 /similarity 내부)
    # 결과를 보기 쉽게 데이터프레임으로 묶기
    results_df = pd.DataFrame({
        'Label': classes,
        'Probability': probs
    }).sort_values(by='Probability', ascending=False)

    # 4. 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./{UPLOAD_DIR}/{now}-{file.filename}"

    # 5. 파일 저장
    await file.seek(0)
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 성공적으로 분석하고 저장했습니다.",
        "filename": file_name,
        "time": now,
        "similarity": results_df.to_dict(orient="records")
    }