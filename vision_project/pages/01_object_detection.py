import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

# 페이지 설정 (가장 상단에 위치)
st.set_page_config(page_title="AI Object Detector", page_icon="🔍", layout="wide")

# 1. 모델 로드 (캐싱 및 에러 핸들링)
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"모델을 불러오는 데 실패했습니다: {e}")
        return None

# 사이드바에서 설정 제어
with st.sidebar:
    st.title("⚙️ 설정")
    conf_threshold = st.slider("신뢰도 임계값 (Confidence)", 0.0, 1.0, 0.25, 0.05)
    st.info("임계값이 낮을수록 더 많은 물체를 찾지만 오검출이 늘어납니다.")

model = load_model("../../models/yolo26m.pt")

# 2. UI 레이아웃
st.title("🔍 Object Detection 실시간 체험")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 이미지 업로드")
    uploaded_file = st.file_uploader(
        label='분석할 이미지를 선택하세요.', 
        type=['png', 'jpg', 'jpeg', 'webp']
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="원본 이미지", use_container_width=True)

# 3. 예측 로직
if uploaded_file:
    with col2:
        st.subheader("📦 분석 결과")
        
        # 버튼 없이 파일 업로드 시 바로 실행되게 하거나, 버튼 유지 가능
        if st.button('분석 시작', use_container_width=True, type="primary"):
            with st.spinner('AI가 이미지를 분석 중입니다...'):
                start_time = time.time()
                
                # 예측 수행
                img = Image.open(uploaded_file)
                results = model.predict(source=img, conf=conf_threshold, save=False)
                
                end_time = time.time()
                inference_time = end_time - start_time

                # 결과 렌더링
                res_plotted = results[0].plot()[:, :, ::-1] # RGB 변환
                st.image(res_plotted, caption=f"분석 완료 ({inference_time:.2f}초)", use_container_width=True)

                # 4. 데이터 요약 및 시각화
                st.markdown("### 📊 검출 요약")
                
                # 검출된 객체 카운팅
                classes = results[0].names
                detected_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                
                if len(detected_indices) > 0:
                    counts = pd.Series([classes[i] for i in detected_indices]).value_counts()
                    
                    # 깔끔한 테이블 형태로 표시
                    df_counts = pd.DataFrame({"객체명": counts.index, "개수": counts.values})
                    st.table(df_counts)
                    
                    # 추가 정보 (Expander)
                    with st.expander("상세 좌표 데이터(Raw Data) 보기"):
                        st.dataframe(results[0].boxes.data.cpu().numpy())
                else:
                    st.info("검출된 객체가 없습니다. 임계값을 낮춰보세요.")

else:
    with col2:
        st.info("이미지를 업로드하면 우측에 분석 결과가 표시됩니다.")

# 푸터
st.divider()
st.caption("Powered by Ultralytics YOLO & Streamlit")