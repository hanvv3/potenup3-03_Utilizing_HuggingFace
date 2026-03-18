import streamlit as st

# 1. Page Config (넓은 화면, 스크롤 최소화)
st.set_page_config(
    page_title="Segmentation Guide",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed" # 한 화면 집중을 위해 사이드바 숨김
)

# 2. Compact Liquid Glass CSS
st.markdown("""
<style>
/* 전체 화면 꽉 차게 패딩 조절 (스크롤 방지) */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px !important;
}

/* 배경 그라데이션 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    overflow-y: hidden; /* 세로 스크롤 숨김 */
}

/* 기본 폰트 설정 (색상은 강제하지 않고 폰트만 애플 스타일로 변경!) */
html, body, .hero-title, .hero-subtitle, .card-title, .card-desc, .hl {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif !important;
}

/* 메인 타이틀 */
.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #1d1d1f 0%, #555555 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 500;
    color: #6e6e73;
    margin-bottom: 2.5rem;
}

/* 2x2 그리드용 유리 질감 카드 */
.glass-card {
    background: rgba(255, 255, 255, 0.45);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    height: 100%;
    min-height: 240px; /* 카드 높이 통일 */
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.55);
}

.card-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
}

.card-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #007AFF;
    margin-bottom: 0.8rem;
}

.card-desc {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #333336;
    font-weight: 500;
    word-break: keep-all;
}

/* 강조 형광펜 효과 */
.hl {
    color: #000;
    font-weight: 700;
    background: linear-gradient(120deg, rgba(0,122,255,0.15) 0%, rgba(0,122,255,0.15) 100%);
    background-repeat: no-repeat;
    background-size: 100% 40%;
    background-position: 0 80%;
    padding: 0 2px;
}
</style>
""", unsafe_allow_html=True)

# 3. Page Header
st.markdown('<div class="hero-title">Image Segmentation 101</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">픽셀 단위로 쪼개서 이해하는 이미지 분할 가이드</div>', unsafe_allow_html=True)

# 4. 2x2 Dashboard Layout
# 첫 번째 행
row1_col1, row1_col2 = st.columns(2, gap="large")

with row1_col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">🧩</div>
        <div class="card-title">모델의 역할</div>
        <div class="card-desc">
            Image segmentation은 입력된 이미지의 <span class="hl">모든 픽셀을 개별적으로 분석</span>하여 각 픽셀이 <span class="hl">어떤 객체나 배경에 속하는지 분류</span>하는 모델입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">📥</div>
        <div class="card-title">입력과 출력</div>
        <div class="card-desc">
            <span class="hl">시각 데이터</span>를 입력하면 단순한 네모 상자가 아닌, <span class="hl">객체의 정확한 외곽선을 따라 칠해진 픽셀 단위의 분할 맵(Segmentation Map) 또는 마스크(Mask)</span>를 내뱉습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# 두 번째 행
row2_col1, row2_col2 = st.columns(2, gap="large")

with row2_col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">💡</div>
        <div class="card-title">무엇을 할 수 있나요?</div>
        <div class="card-desc">
            Segmentation으로 <span class="hl">자율주행 차량이 주행 가능한 도로와 인도를 정밀하게 파악하거나, 의료 영상에서 종양의 정확한 크기/모양을 추출하고, 화상회의 배경을 분리(누끼 따기)</span>할 수 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">⚠️</div>
        <div class="card-title">학습 시 주의사항</div>
        <div class="card-desc">
            학습 시에는 <span class="hl">픽셀 단위의 정교한 라벨링(Annotation)에 드는 막대한 시간과 비용, 그리고 넓은 배경과 좁은 객체 픽셀 간의 극심한 클래스 불균형(Class Imbalance)</span>을 주의해야 합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)