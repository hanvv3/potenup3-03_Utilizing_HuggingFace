import streamlit as st
import requests
import pandas as pd
from PIL import Image
import time

# 1. Page Config (가장 상단에 위치해야 합니다)
st.set_page_config(page_title="Zero-shot Classification with CLIP", page_icon="🔍", layout="wide")

# 🔗 FastAPI 백엔드 URL 설정 (포트나 주소가 다르다면 맞춰서 수정해주세요)
API_URL = "http://127.0.0.1:8080/similarity"

# 2. 사이드바 설정 통합
with st.sidebar:
    st.title("⚙️ 설정 및 프롬프트")
    conf_threshold = st.slider("신뢰도 임계값 (Confidence)", 0.0, 1.0, 0.25, 0.05)
    st.info("임계값이 낮을수록 더 많은 물체를 찾지만 오검출이 늘어납니다.")
    
    # CLIP은 'a photo of a ~' 포맷일 때 성능이 더 좋습니다.
    text_prompt = st.text_area(
        "찾을 객체 입력 (영어로 입력, 쉼표 구분)", 
        value="a photo of a person, a photo of a backpack, a photo of a dog, a photo of a car"
    )

# 3. UI 레이아웃
st.title("🔍 CLIP Model 실시간 체험 (FastAPI 연동)")
st.markdown("**Zero-shot Classification**: 학습된 적 없는 클래스라도 텍스트 설명만으로 이미지를 분류해냅니다.")
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 이미지 업로드")
    uploaded_file = st.file_uploader(
        label='분석할 이미지를 선택하세요.', 
        type=['png', 'jpg', 'jpeg', 'webp']
    )
    
    if uploaded_file:
        # 이미지를 화면에 보여주기 위해 읽기
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="원본 이미지", use_container_width=True)

# 4. 백엔드 통신 및 시각화 로직
if uploaded_file and text_prompt:
    with col2:
        st.subheader("📦 분석 결과")
        
        if st.button('🚀 분석 시작', use_container_width=True, type="primary"):
            with st.spinner('FastAPI 서버에서 이미지를 분석 중입니다...'):
                start_time = time.time()
                
                try:
                    # 💡 FastAPI로 보낼 파일과 데이터 포장
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    data = {
                        "text": text_prompt
                    }
                    
                    # 💡 FastAPI 서버로 POST 요청 전송
                    response = requests.post(API_URL, files=files, data=data)
                    
                    # 응답이 정상(200 OK)일 때만 처리
                    if response.status_code == 200:
                        result_json = response.json()
                        
                        # FastAPI가 딕셔너리 리스트로 보낸 데이터를 다시 DataFrame으로 복원
                        similarity_data = result_json.get("similarity", [])
                        results_df = pd.DataFrame(similarity_data)
                        
                        end_time = time.time()
                        inference_time = end_time - start_time

                        if not results_df.empty:
                            # 5. 데이터 요약 및 시각화 렌더링
                            top_label = results_df.iloc[0]['Label']
                            top_prob = float(results_df.iloc[0]['Probability'])
                            
                            # 가장 높은 확률을 가진 결과 출력 (임계값 비교)
                            if top_prob >= conf_threshold:
                                st.success(f"**가장 유력한 결과:** `{top_label}` ({top_prob:.1%} 일치)")
                            else:
                                st.warning(f"설정하신 임계값({conf_threshold:.2f})을 넘는 확신 있는 결과가 없습니다. (최고: {top_label}, {top_prob:.1%})")

                            st.markdown("### 📊 확률 분포 요약")
                            
                            # 각 레이블별 확률을 프로그레스 바(Progress bar)로 시각화
                            for idx, row in results_df.iterrows():
                                label = row['Label']
                                prob = float(row['Probability'])
                                
                                p_col1, p_col2 = st.columns([1, 3])
                                with p_col1:
                                    st.write(f"**{label}**")
                                with p_col2:
                                    st.progress(prob, text=f"{prob:.1%}")

                            st.caption(f"⏱️ 통신 및 추론 시간: {inference_time:.3f} 초")
                        else:
                            st.warning("분석 결과가 없습니다.")
                    
                    else:
                        st.error(f"서버 에러가 발생했습니다. (상태 코드: {response.status_code})")
                
                except requests.exceptions.ConnectionError:
                    st.error("🚨 FastAPI 서버에 연결할 수 없습니다. 터미널에서 `uvicorn` 서버가 실행 중인지 확인해주세요!")

else:
    with col2:
        st.info("👈 왼쪽에서 이미지를 업로드하고 프롬프트를 입력하면 분석 결과가 표시됩니다.")

# 푸터
st.divider()
st.caption("Powered by FastAPI & Streamlit | OpenAI CLIP")