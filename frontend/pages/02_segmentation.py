import streamlit as st
from ultralytics.models.sam import SAM3SemanticPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 페이지 설정 ---
st.set_page_config(page_title="Pure SAM3 Text Segmenter", page_icon="🧩", layout="wide")

# --- 1. SAM 3 전용 Predictor 로드 (캐싱) ---
@st.cache_resource
def load_sam3(model_path):
    try:
        overrides = dict(
            task="segment", 
            mode="predict", 
            model=model_path, 
            conf=0.25 # 초기 기본값
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        return predictor
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

# 사용자 경로에 맞게 모델 로드
predictor = load_sam3("../models/sam3.pt")

# --- 2. 제공된 시각화 함수 ---
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

# --- 3. UI 레이아웃 ---
st.title("🧩 Pure SAM3 Text-Prompt Segmentation")
st.markdown("**오직 SAM 3 단독**으로 텍스트 프롬프트를 읽고 객체를 분할합니다.")
st.divider()

with st.sidebar:
    st.title("⚙️ 설정 및 프롬프트")
    text_prompt = st.text_area(
        "찾을 객체 입력 (영어로 입력, 쉼표 구분)", 
        value="person, backpack, dog, car"
    )
    conf_threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.25, 0.05)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 이미지 업로드")
    uploaded_file = st.file_uploader(label='이미지 선택', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_file:
        st.image(uploaded_file, caption="원본 이미지", use_container_width=True)

# --- 4. 실행 로직 ---
if uploaded_file and text_prompt:
    if st.button('SAM3 텍스트 세그멘테이션 시작', use_container_width=True, type="primary"):
        with st.spinner('SAM3 단독 모델이 텍스트를 기반으로 분석 중입니다...'):
            start_time = time.time()
            
            # 💡 [핵심 포인트] 입력된 텍스트 프롬프트를 콤마 기준으로 분리하여 '리스트'로 만듭니다.
            target_classes = [x.strip() for x in text_prompt.split(',') if x.strip()]
            
            # 이미지를 BGR Numpy 배열로 변환
            img_pil = Image.open(uploaded_file).convert("RGB")
            img_bgr = np.array(img_pil)[:, :, ::-1] 
            
            # 슬라이더에서 받은 conf 값을 동적으로 덮어쓰기
            predictor.args.conf = conf_threshold
            
            # 1. 이미지를 Predictor에 세팅
            predictor.set_image(img_bgr)
            
            # 2. 💡 [치명적 에러 해결] texts가 아니라 'text' 인자이며, 리스트(target_classes)를 전달합니다!
            results = predictor(text=target_classes)
            
            result = results[0]
            end_time = time.time()

            with col2:
                st.subheader("🖼️ 분석 결과")
                st.success(f"분석 시간: {end_time - start_time:.2f}초")

                # 전체 세그멘테이션 결과 시각화
                res_plotted = result.plot()[:, :, ::-1]
                st.image(res_plotted, caption="전체 세그멘테이션 결과", use_container_width=True)

                st.divider()
                st.markdown("### ✂️ 개별 객체 추출 (투명 배경)")

                fig_objects = extract_and_plot_objects(result)
                if fig_objects:
                    st.pyplot(fig_objects)
                else:
                    st.warning("입력한 텍스트에 해당하는 객체를 찾지 못했습니다. 임계값을 조절해보세요.")

st.divider()
st.caption("Powered by Pure SAM3 & Streamlit.")