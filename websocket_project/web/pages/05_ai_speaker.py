import requests
import streamlit as st

SERVER = "http://localhost:8888"

st.title("AI 스피커")
st.caption("녹음 → STT → AI 답변 → TTS → 재생")

# ── 녹음 ──────────────────────────────────────
audio = st.audio_input("녹음하기")

if st.button("전송", key="btn") and audio:
    # STEP1. STT — 음성 → 텍스트
    res = requests.post(f"{SERVER}/stt", files={"audio": audio})
    question = res.json()["result"]
    st.session_state["question"] = question

    # STEP2. AI 답변 — 텍스트 → 텍스트
    res = requests.post(f"{SERVER}/chat", json={"question": question})
    answer = res.json()["result"]
    st.session_state["answer"] = answer

    # STEP3. TTS — 텍스트 → 오디오
    res = requests.post(f"{SERVER}/tts", json={"text": answer})
    st.session_state["audio"] = res.content

# ── 결과 표시 ──────────────────────────────────
if st.session_state.get("question"):
    st.subheader("내가 한 말")
    with st.container(border=True):
        st.write(st.session_state["question"])

if st.session_state.get("answer"):
    st.subheader("AI 답변")
    with st.container(border=True):
        st.write(st.session_state["answer"])

if st.session_state.get("audio"):
    st.subheader("AI 음성")
    st.audio(st.session_state["audio"], format="audio/mp3", autoplay=True)
