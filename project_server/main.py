# (경로가 project_server에 있을 때) uvicorn main:app --port 8080 --reload
from fastapi import FastAPI, UploadFile, File
import shutil
from datetime import datetime

import io
from PIL import Image

# 모델 불러오기
from ultralytics import YOLO
model = YOLO("../models/yolo26m.pt")
print("모델을 불러왔습니다.")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 사용자가 이미지를 입력하면 서버는 이미지를 저장한다.
# 파일 저장 이름 바꾸기(오늘날짜-파일이름) hint: datetime
@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"../images/{now}-{file.filename}"

    # 파일 저장
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": datetime.now().strftime("%Y%m%d%H%M%S")
    }

@app.post("/detect_image")
async def predict_yolo(file: UploadFile = File(...)):
    # 1. img 읽기
    file_bytes = await file.read()
    img = Image.open(io.BytesIO(file_bytes))

    # 2. 예측하기
    results = model.predict(img)
    result = results[0]

    # 3. 데이터 만들기
    detections = []
    names = result.names 
    for x1, y1, x2, y2, conf, label_idx in result.boxes.data:
        detections.append(
            {
                "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                "confidence": conf.item(),  # ✨ 수정됨: Tensor를 float으로 변환!
                "label": names[int(label_idx)]
            }
        )

    # 4. 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"../images/{now}-{file.filename}"

    # 5. 파일 저장
    await file.seek(0)  # ✨ 수정됨: 닫힌 커서를 다시 맨 앞으로 되돌림!
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 성공적으로 분석하고 저장했습니다.",
        "filename": file_name,
        "time": now,
        "object_detection": detections
    }