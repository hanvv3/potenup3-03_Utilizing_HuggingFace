import streamlit as st
import requests

API_URL = "http://127.0.0.1:8080/summarize_audio"

st.set_page_config(page_title="회의 음성 요약기", page_icon="🎙️", layout="wide")

st.title("🎙️ 회의 녹음 요약기")
st.write("음성 파일을 업로드하면 회의 내용을 markdown 형식으로 정리해줍니다.")

uploaded_file = st.file_uploader(
    "음성 파일 업로드",
    type=["mp3", "wav", "m4a", "mpeg", "mp4", "webm"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("회의록 생성"):
        with st.spinner("음성을 전사하고 회의록을 생성하는 중입니다..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream"
                )
            }

            try:
                response = requests.post(API_URL, files=files, timeout=300)

                if response.status_code == 200:
                    result = response.json()

                    st.success("회의록 생성 완료")

                    st.subheader("Markdown 결과")
                    st.markdown(result["markdown"])

                    st.download_button(
                        label="회의록 .md 다운로드",
                        data=result["markdown"],
                        file_name="meeting_summary.md",
                        mime="text/markdown"
                    )

                    with st.expander("전사 원문 보기"):
                        st.text_area(
                            "Transcript",
                            value=result["transcript"],
                            height=300
                        )

                else:
                    st.error(f"API 오류: {response.status_code}")
                    try:
                        st.json(response.json())
                    except Exception:
                        st.text(response.text)

            except requests.exceptions.Timeout:
                st.error("요청 시간이 너무 오래 걸렸습니다.")
            except Exception as e:
                st.error(f"오류 발생: {e}")