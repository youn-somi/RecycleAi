# streamlit_app.py — 배포용 화면 (독립 실행형)

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 웹페이지 설정
st.set_page_config(
    page_title="RecycleAI — 분리수거 AI",
    page_icon="♻️",
    layout="centered"
)

# 기존 디자인의 색상과 모양 적용
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef2f3 0%, #8e9eab 100%);
    }

    .block-container {
        max-width: 500px;
        padding-top: 3rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    }

    h1 {
        color: #1a365d !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        text-align: center;
    }

    .subtitle {
        color: #64748b;
        font-size: 14px;
        text-align: center;
        margin-bottom: 24px;
    }

    div[data-testid="stFileUploader"] section {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        background: #f8fafc;
        padding: 16px;
    }

    div.stButton > button {
        width: 100%;
        background: #0284c7;
        color: white;
        border: none;
        padding: 14px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #0369a1;
        color: white;
    }

    div[data-testid="stAlert"] {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# AI 모델 캐시 로드 (recycle_model.keras 파일 직접 불러오기)
@st.cache_resource
def load_ai_model():
    return load_model("recycle_model.keras")

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"모델 로드 중 오류 발생: {e}")

# 클래스 이름 정의 (학습시킨 순서에 맞게 수정 필요시 변경)
CLASS_NAMES = ['종이상자', '유리', '금속', '종이', '플라스틱', '일반쓰레기']  # ✅

def predictphoto(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))  # 모델 크기에 맞춤
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    
    if predicted_class_idx < len(CLASS_NAMES):
        result_name = CLASS_NAMES[predicted_class_idx]
    else:
        result_name = f"클래스 {predicted_class_idx}"
        
    return result_name, round(confidence, 2)

# 웹페이지 화면 구성
with st.container(border=True):

    st.markdown("<h1>♻️ RecycleAI</h1>", unsafe_allow_html=True)

    st.markdown(
        '<p class="subtitle">쓰레기 사진을 올리면 AI가 종류를 분석해 드립니다</p>',
        unsafe_allow_html=True
    )

    # 사진 업로드
    uploaded_file = st.file_uploader(
        "📂 클릭하여 사진 파일 선택하기",
        type=["jpg", "jpeg", "png"]
    )

    # 사진 미리보기
    if uploaded_file is not None:

        uploaded_file.seek(0)
        preview = Image.open(uploaded_file).convert("RGB")
        st.image(preview, use_container_width=True)

        # AI 분류 실행 버튼
        if st.button("분류하기"):

            uploaded_file.seek(0)

            with st.spinner("AI가 분석 중입니다..."):
                result, prob = predictphoto(uploaded_file)

            st.success(f"AI 분석 결과: **{result}** ({prob}%)")