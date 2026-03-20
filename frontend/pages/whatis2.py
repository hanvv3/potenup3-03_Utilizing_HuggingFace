import streamlit as st

# 1. Page Config
st.set_page_config(
    page_title="CLIP Guide",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Compact Liquid Glass CSS
st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px !important;
}

.stApp {
    background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    overflow-y: hidden;
}

html, body, .hero-title, .hero-subtitle, .card-title, .card-desc, .hl {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif !important;
}

.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #2c3e50 0%, #4b7bec 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 500;
    color: #4b6584;
    margin-bottom: 2.5rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
    height: 100%;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.65);
}

.card-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
}

.card-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4b7bec;
    margin-bottom: 0.8rem;
}

.card-desc {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #2f3542;
    font-weight: 500;
    word-break: keep-all;
}

.hl {
    color: #000;
    font-weight: 700;
    background: linear-gradient(120deg, rgba(75, 123, 236, 0.2) 0%, rgba(75, 123, 236, 0.2) 100%);
    background-repeat: no-repeat;
    background-size: 100% 40%;
    background-position: 0 80%;
    padding: 0 2px;
}
</style>
""", unsafe_allow_html=True)

# 3. Page Header
st.markdown('<div class="hero-title">CLIP: Bridge the Gap</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">언어와 시각 정보를 하나의 공간에서 연결하는 혁신적 멀티모달 모델</div>', unsafe_allow_html=True)

# 4. 2x2 Dashboard Layout
row1_col1, row1_col2 = st.columns(2, gap="large")

with row1_col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">🔗</div>
        <div class="card-title">모델의 역할</div>
        <div class="card-desc">
            CLIP은 <span class="hl">이미지와 텍스트를 동일한 벡터 공간에 매핑</span>합니다. 수많은 이미지와 그를 설명하는 텍스트 쌍을 대조 학습(Contrastive Learning)하여, <span class="hl">시각 정보와 언어의 연관성을 인간처럼 이해</span>합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">🧠</div>
        <div class="card-title">핵심 특징: Zero-shot</div>
        <div class="card-desc">
            특정 레이블 없이도 <span class="hl">처음 보는 데이터셋에 대해 분류를 수행</span>할 수 있습니다. 미리 정의된 클래스 대신 "사진 속의 개"와 같은 <span class="hl">자연어 설명만으로 정답을 찾아내는 유연함</span>이 가장 큰 특징입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns(2, gap="large")

with row2_col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">🚀</div>
        <div class="card-title">활용 분야</div>
        <div class="card-desc">
            텍스트로 이미지를 찾는 <span class="hl">시맨틱 이미지 검색</span>, Stable Diffusion 같은 <span class="hl">생성 AI의 텍스트 조건 입력부</span>, 별도의 추가 학습 없는 <span class="hl">이미지 분류기(Zero-shot Classifier)</span> 구축에 널리 활용됩니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-icon">⚖️</div>
        <div class="card-title">한계점과 과제</div>
        <div class="card-desc">
            이미지의 <span class="hl">미세한 차이나 수치 정보(객체의 개수 등) 파악</span>에는 약점을 보일 수 있습니다. 또한, 학습 데이터에 포함된 <span class="hl">사회적 편향(Bias)이 모델에 그대로 반영</span>될 수 있어 주의가 필요합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)