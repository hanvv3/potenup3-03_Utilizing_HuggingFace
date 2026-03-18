# (경로가 project_server에 있을 때) uvicorn main:app --port 8080 --reload
from fastapi import FastAPI, UploadFile, File, Request
from contextlib import asynccontextmanager

import matplotlib.pyplot as plt
import shutil
import os
import io
from datetime import datetime
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor

# -----------------------
# 설정
# -----------------------
UPLOAD_DIR = "upload_img"
ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔥 startup 영역
    print("============= Model loading initiated =============")
    overrides = dict(
        task='segment',
        mode='predict',
        model='../models/sam3.pt',
        conf=0.25
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    app.state.predictor = predictor   # ✅ 여기에 저장

    print("================== Model Ready ====================")

    yield   # ----------------- 여기까지가 startup -----------------

    # 🔥 shutdown 영역 (필요하면 사용)
    print("Shutting down...")

def extract_and_plot_objects(result):
    if result.masks is None or len(result.masks.data) == 0:
        return None

    extracted_images = []
    
    # 원본 이미지 가져오기 (BGR을 RGB로 변환)
    orig_img_rgb = Image.fromarray(result.orig_img[:, :, ::-1]).convert('RGBA')
    orig_w, orig_h = orig_img_rgb.size

    for idx, mask_tensor in enumerate(result.masks.data):
        mask_np = mask_tensor.cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_resized = mask_img.resize((orig_w, orig_h), Image.NEAREST)

        obj_img = orig_img_rgb.copy()
        obj_img.putalpha(mask_resized)
        
        x1, y1, x2, y2 = result.boxes.xyxy[idx].int().tolist()
        crop_img = obj_img.crop((x1, y1, x2, y2))
        extracted_images.append(crop_img)

    num_objects = len(extracted_images)
    cols = min(4, num_objects)
    rows = (num_objects + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if num_objects == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # SAM3는 텍스트 프롬프트를 인식하면 names에 해당 텍스트를 저장합니다.
    names = result.names
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for i, img in enumerate(extracted_images):
        axes[i].imshow(img)
        class_name = names[classes[i]] if classes[i] in names else f"Object {i+1}"
        axes[i].set_title(f"{class_name}", fontsize=10)
        axes[i].axis('off')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return fig

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