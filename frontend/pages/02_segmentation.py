import streamlit as st
import requests
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io

# --- 페이지 설정 ---
st.set_page_config(page_title="SAM3 API Segmenter", page_icon="🧩", layout="wide")

API_URL = "http://127.0.0.1:8080/detect_image"

# --- 💡 JSON 데이터를 기반으로 객체를 추출하는 함수 ---
def extract_and_plot_objects_from_json(orig_img_pil, detections):
    if not detections:
        return None

    # 투명도(Alpha) 채널을 위해 RGBA로 변환
    orig_img_rgba = orig_img_pil.convert('RGBA')
    extracted_images = []
    labels = []

    for det in detections:
        box = det['box']          # [x1, y1, x2, y2]
        polygon = det['polygon']  # [[x, y], [x, y], ...]
        label = det['label']

        # 1. 원본 이미지와 동일한 크기의 빈(검은색, 투명도 0) 마스크 생성
        mask = Image.new('L', orig_img_rgba.size, 0)
        draw = ImageDraw.Draw(mask)

        # 2. 서버에서 받은 폴리곤 좌표로 하얀색(255) 다각형 그리기
        if polygon and len(polygon) > 2:
            poly_tuples = [tuple(pt) for pt in polygon]
            draw.polygon(poly_tuples, outline=255, fill=255)
        else:
            # 폴리곤 데이터가 부족하면 바운딩 박스로 대체
            draw.rectangle(box, outline=255, fill=255)

        # 3. 원본 이미지에 마스크 씌우기 (마스크가 흰색인 곳만 보임)
        obj_img = orig_img_rgba.copy()
        obj_img.putalpha(mask)

        # 4. 바운딩 박스 크기만큼 잘라내기 (Crop)
        crop_img = obj_img.crop((box[0], box[1], box[2], box[3]))
        
        extracted_images.append(crop_img)
        labels.append(label)

    # 5. 시각화 (Matplotlib)
    num_objects = len(extracted_images)
    cols = min(4, num_objects)
    rows = (num_objects + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if num_objects == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(extracted_images):
        axes[i].imshow(img)
        axes[i].set_title(labels[i], fontsize=10)
        axes[i].axis('off')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return fig

# --- UI 레이아웃 ---
st.title("🧩 SAM3 Segmentation API Client")
st.markdown("**FastAPI 서버**로 이미지를 보내고 텍스트 프롬프트를 기반으로 객체를 분할합니다.")
st.divider()

with st.sidebar:
    st.title("⚙️ 설정 및 프롬프트")
    text_prompt = st.text_area(
        "찾을 객체 입력 (영어로 입력, 쉼표 구분)", 
        value="person, backpack, dog, car"
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 이미지 업로드")
    uploaded_file = st.file_uploader(label='이미지 선택', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_file:
        orig_img_pil = Image.open(uploaded_file)
        st.image(orig_img_pil, caption="원본 이미지", use_container_width=True)

# --- 실행 로직 ---
if uploaded_file and text_prompt:
    if st.button('SAM3 서버로 전송 및 분석 시작', use_container_width=True, type="primary"):
        with st.spinner('FastAPI 서버에서 이미지를 분석 중입니다...'):
            start_time = time.time()
            
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {"text": text_prompt}
                
                response = requests.post(API_URL, files=files, data=data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result_data = response.json()
                    detections = result_data.get("object_detection", [])
                    
                    with col2:
                        st.subheader("🖼️ 분석 결과")
                        st.success(f"서버 응답 시간: {end_time - start_time:.2f}초")
                        
                        if detections:
                            st.write(f"총 {len(detections)}개의 객체를 추출했습니다!")
                            
                            st.markdown("### ✂️ 개별 객체 추출 (투명 배경)")
                            # 💡 수정한 함수 호출
                            fig_objects = extract_and_plot_objects_from_json(orig_img_pil, detections)
                            
                            if fig_objects:
                                st.pyplot(fig_objects)
                        else:
                            st.warning("입력한 텍스트에 해당하는 객체를 찾지 못했습니다.")
                            
                else:
                    st.error(f"서버 에러 발생 (상태 코드: {response.status_code})")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

st.divider()
st.caption("Powered by FastAPI & Streamlit.")