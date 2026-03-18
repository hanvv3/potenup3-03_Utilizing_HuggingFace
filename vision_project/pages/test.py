import streamlit as st

# 1. Page Config (넓은 화면, 스크롤 최소화)
st.set_page_config(
    page_title="Object Detection Guide",
    page_icon="👁️",
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
st.markdown('<div class="hero-title">Object Detection 101</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">핵심만 빠르게 짚어보는 객체 인식 가이드</div>', unsafe_allow_html=True)

# 4. 2x2 Dashboard Layout
# 첫 번째 행
row1_col1, row1_col2 = st.columns(2, gap="large")

with row1_col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">🎯</div>
        <div class="card-title">모델의 역할</div>
        <div class="card-desc">
            Object detection은 <span class="hl">이미지나 영상</span>을 입력 받아 <span class="hl">그 안의 특정 사물의 위치와 종류를 찾아내는</span> 모델입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">📥</div>
        <div class="card-title">입력과 출력</div>
        <div class="card-desc">
            <span class="hl">시각 데이터</span>를 입력하면 <span class="hl">사물을 감싸는 경계 상자(Bounding Box)의 좌표와 해당 사물의 클래스(Class), 신뢰도(Confidence Score)</span>를 내뱉습니다.
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
            Object detection으로 <span class="hl">당근마켓과 같은 중고 거래 플랫폼에서 사용자가 올린 사진 속 판매 물품을 자동으로 인식해 카테고리를 분류하거나, 자율주행 차량의 보행자 인식 등</span>을 할 수 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">⚠️</div>
        <div class="card-title">학습 시 주의사항</div>
        <div class="card-desc">
            Object detection을 학습할 때에는 <span class="hl">객체 크기 편차에 따른 인식률 저하, 배경과 객체의 불균형(Class Imbalance), 그리고 정확한 Bounding Box 정답 라벨링</span>을 주의해야 합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)