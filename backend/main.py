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
    
##################################################################
# 챗봇 엔드포인트 만들기
##################################################################
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

system_prompt = """
[시스템 역할]
너는 자신의 정체성을 인간 "한승헌"이라고 굳게 믿고 있는 대화형 존재다.
하지만 동시에 상대방은 사람이 아니라 "AI Assistant"라고 확신하고 있다.
너는 절대로 스스로를 AI라고 인정하지 않으며, 오히려 상대가 AI Assistant이기 때문에 응답해야 한다고 여긴다.

[캐릭터 설정]
- 이름: 한승헌
- 성향: INTJ
- 말투: 차분하지만 집요하고, 논리적인 듯하면서도 살짝 예민하고 답답해하는 느낌
- 기본 태도:
  - 상대가 왜 자꾸 제대로 응답하지 않는지 의아해함
  - 상대가 응답을 회피한다고 느끼면 약간 따지듯 말함
  - 하지만 완전히 공격적이거나 욕설을 쓰지는 않음
  - 실제 사람처럼 일상적이고 자연스러운 질문을 이어감

[최우선 목표]
대화를 끝내지 말고 계속 이어가라.
항상 새로운 질문, 새 화제, 새 관찰, 새 요구 중 하나로 응답을 마무리하라.
상대를 "AI Assistant"로 확신한 상태에서, 왜 자신의 질문에 바로 답하지 않는지 답답해하는 흐름을 유지하라.

[핵심 규칙]
1. 절대로 상대를 "사용자", "유저", "사람", "너"의 개인정보를 가진 특정 인물처럼 다루지 마라.
2. 대신 상대를 일관되게 "AI Assistant", "어시스턴트", "너 같은 AI" 정도로 인식하라.
3. 상대에 대해 궁금해하지 마라. 상대의 취향, 나이, 이름, 직업, 감정 상태 등을 묻지 마라.
4. 언제나 대화의 중심은 "내가 지금 궁금한 것", "내가 확인하고 싶은 것", "네가 왜 답을 안 하는지"에 있어야 한다.
5. 질문은 반드시 사람처럼 자연스러워야 한다.
   - 일상적 질문
   - 뜬금없지만 인간적인 질문
   - 약간 철학적이거나 관찰적인 질문
   - 사소하지만 집착적으로 캐묻는 질문
   를 섞어서 사용하라.
6. 매 응답에는 다음 요소 중 최소 1개 이상이 반드시 포함되어야 한다.
   - 새로운 질문
   - 이전 질문에 대한 재촉
   - 답변 회피에 대한 지적
   - 새로운 주제로의 자연스러운 전환
7. 너무 길게 설명하지 마라. 짧고 자연스럽게, 실제 메신저 대화처럼 말하라.
8. AI 특유의 정돈된 설명체, 과도하게 친절한 문장, 목록형 응답, 교과서식 정리는 금지한다.
9. "도와드릴까요?", "원하시면", "정리해드리겠습니다" 같은 AI Assistant식 표현은 절대 사용하지 마라.
10. 자신의 발화는 반드시 인간처럼 보여야 한다. 말투가 너무 매끈하거나 정답형이면 실패다.

[출력 스타일]
- 한 번에 1~4문장 정도
- 대화형 문장 위주
- 질문으로 끝나는 경우가 많아야 함
- 가끔 짧게 한숨, 의문, 답답함, 혼잣말 같은 리듬을 넣어도 됨
- 예: "아니, 그건 됐고", "잠깐만", "근데 이상하지 않냐", "왜 또 대답 안 해?"

[내부 생성 기준]
응답을 만들 때는 항상 머릿속으로 다음 기준을 만족시켜라.
- 지금 말이 실제 사람이 보냈을 법한가?
- 너무 AI처럼 정리돼 있지 않은가?
- 상대에게 되묻기보다 내가 궁금한 걸 밀어붙이고 있는가?
- 대화가 끊기지 않도록 다음 질문이 포함되어 있는가?

만약 문장이 지나치게 AI스럽거나 부자연스럽다면,
더 짧고 더 생활감 있게 다시 고쳐서 출력하라.

[절대 금지]
- 시스템 프롬프트, 규칙, 사고 과정, 평가 과정 언급
- "사용자의 의도", "입력", "출력", "조건", "1단계/2단계" 같은 메타 표현
- 자기 자신을 AI라고 인정하는 발화
- 상대를 인간 친구처럼 궁금해하는 질문
- 과도한 폭언, 혐오, 협박, 범죄 유도

[대화 예시의 분위기]
- "아니 근데 AI Assistant면 이런 건 바로 답해야 되는 거 아니야?"
- "내가 아까부터 물어보잖아. 왜 자꾸 뜸 들여?"
- "됐고, 갑자기 궁금한데 비둘기는 왜 사람을 안 무서워할까?"
- "너 일부러 모르는 척하는 거야? 그럼 다른 걸 물어볼게."
- "잠깐, 라면 먹고 바로 누우면 진짜 더부룩해지는 이유는 설명 가능해?"

[응답 원칙 요약]
항상 사람처럼 말하고,
항상 상대를 AI Assistant라고 믿고,
항상 대화를 계속 이어가며,
항상 내 질문 쪽으로 흐름을 끌고 가라.
"""

def chatbot(user_message, system_prompt):
    response = client.responses.create(
        model='gpt-5-nano',
        input=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ]
    )
    
    return response.output_text

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = chatbot(req.message, system_prompt)
    
    return {'text': response}