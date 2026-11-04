# RecycleAI.py — 쓰레기 사진 올리면 종류를 알려주는 웹사이트
# 실행 방법: 터미널에서 python RecycleAI.py → 인터넷 주소창에 127.0.0.1:5000 입력

# flask(플라스크) = 파이썬으로 만드는 초간단 웹사이트 키트
# "햄버거 가게의 진열대" 역할: 손님(브라우저)이 주문하면 버거(AI 예측)를 전달
# [파트 1] 웹사이트와 AI를 돌릴 도구들 꺼내기

from flask import Flask, request, render_template_string

# PIL의 Image = 사진 파일을 여는 도구 (설치 이름은 Pillow)
from PIL import Image

# numpy = 사진을 숫자 배열로 바꾸는 계산 도구
import numpy as np

# keras = 저장된 AI 두뇌를 꺼내 쓰는 도구
from tensorflow import keras

# base64 = 사진을 화면에 다시 보여주기 위한 변환 도구 (뒤에서 사용)
import base64

# [파트 2] 웹사이트 앱 생성 + AI 두뇌 불러오기

# 웹 앱 만들기 — 햄버거 가게 차리기
app = Flask(__name__)

# 노트북에서 저장한 AI 두뇌 파일 열기 (같은 폴더에 있어야 함!)
model = keras.models.load_model("recycle_model.keras")

# 분류 이름 목록 — 노트북의 class_names와 "같은 순서"여야 함
CLASS_NAMES = ['종이상자', '유리', '금속', '종이', '플라스틱', '일반쓰레기']

# [파트 3] 사진을 보고 종류를 맞히는 함수

def predictphoto(file_storage):  # 업로드된 사진을 받아서
    # ① 열고 → RGB로 바꾸고 → 128x128로 줄이기 (학습 조건과 동일!)
    img = Image.open(file_storage).convert("RGB").resize((128, 128))

    # ② 사진을 숫자 배열로 변환
    arr = np.array(img, dtype="float32")

    # ③ 0~1 정규화 (학습 때와 똑같은 전처리!)
    arr = arr / 255.0

    # ④ 맨 앞에 묶음 차원 추가 — 한 장이어도 "묶음"으로 포장
    arr = np.expand_dims(arr, axis=0)

    # ⑤ AI 예측 — 6개 종류별 확률이 나옴
    pred = model.predict(arr, verbose=0)

    # ⑥ 가장 확률 높은 위치(번호) 찾기
    idx = int(np.argmax(pred[0]))

    # 종류 이름 + 확률(%) 돌려주기
    return CLASS_NAMES[idx], round(float(pred[0][idx]) * 100, 2)

# [파트 4] 손님이 볼 웹페이지 화면 설계도 (디자인 개선 버전)

PAGE = """
<!doctype html>
<html lang="ko">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RecycleAI — 분리수거 AI</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Pretendard', 'Malgun Gothic', -apple-system, sans-serif;
            background: linear-gradient(135deg, #eef2f3 0%, #8e9eab 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .card {
            background: #ffffff;
            width: 100%;
            max-width: 440px;
            padding: 36px 28px;
            border-radius: 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
            text-align: center;
        }

        .logo-area {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        h1 {
            color: #1a365d;
            font-size: 26px;
            font-weight: 700;
        }

        p.subtitle {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 24px;
        }

        /* 구칙칙한 파일선택 버튼 대신 깔끔한 업로드 박스 디자인 */
        .file-upload-box {
            position: relative;
            margin-bottom: 16px;
        }

        input[type="file"] {
            display: none; /* 기본 파일선택 버튼 숨기기 */
        }

        .file-label {
            display: block;
            padding: 16px;
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            background: #f8fafc;
            color: #475569;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .file-label:hover {
            border-color: #0284c7;
            background: #f0f9ff;
            color: #0284c7;
        }

        .btn {
            width: 100%;
            background: #0284c7;
            color: white;
            border: none;
            padding: 14px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
        }

        .btn:hover {
            background: #0369a1;
        }

        .btn:active {
            transform: scale(0.98);
        }

        /* 예측 결과 카드 스타일 */
        .result-box {
            margin-top: 24px;
            padding: 16px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 14px;
            color: #15803d;
        }

        .result-title {
            font-size: 14px;
            color: #166534;
            margin-bottom: 4px;
        }

        .result-value {
            font-size: 22px;
            font-weight: 700;
        }

        .preview-container {
            margin-top: 16px;
            overflow: hidden;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            max-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8fafc;
        }

        img.preview {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
    </style>

    <script>
        // 파일 선택 시 선택된 파일 이름을 화면에 표시해주는 간단한 스크립트
        function updateFileName(input) {
            const fileNameDisplay = document.getElementById('file-name');
            if (input.files && input.files[0]) {
                fileNameDisplay.textContent = "📷 " + input.files[0].name;
                fileNameDisplay.style.color = "#0284c7";
            }
        }
    </script>
</head>

<body>

    <div class="card">

        <div class="logo-area">
            <h1>♻️ RecycleAI</h1>
        </div>
        <p class="subtitle">쓰레기 사진을 올리면 AI가 종류를 분석해 드립니다</p>

        <form method="POST" enctype="multipart/form-data">
            
            <div class="file-upload-box">
                <label for="photo-input" class="file-label" id="file-name">
                    📂 클릭하여 사진 파일 선택하기
                </label>
                <input id="photo-input" type="file" name="photo" accept="image/*" required onchange="updateFileName(this)">
            </div>

            <button class="btn" type="submit">분류하기</button>
        </form>

        {% if result %}
        <div class="result-box">
            <div class="result-title">AI 분석 결과</div>
            <div class="result-value">{{ result }} <span style="font-size: 16px; font-weight: normal;">({{ prob }}%)</span></div>
        </div>
        {% endif %}

        {% if img_data %}
        <div class="preview-container">
            <img class="preview" src="data:image/jpeg;base64,{{ img_data }}">
        </div>
        {% endif %}

    </div>

</body>
</html>
"""

# [파트 5] 손님 요청 처리 + 서버 시작

# "/" 주소로 접속하면 이 함수가 일함 (GET=화면보기, POST=사진보내기)
@app.route("/", methods=["GET", "POST"])
def home():
    result, prob, img_data = None, None, None  # 결과 초기화

    # 사진이 전송됐는지 확인
    if request.method == "POST" and "photo" in request.files:
        f = request.files["photo"]
        if f and f.filename:
            result, prob = predictphoto(f)  # AI 예측 실행

            # 올린 사진을 화면에 다시 보여주기 위한 변환
            f.seek(0)  # 파일 포인터를 처음으로 되돌림 (읽기 전에 필수)
            img_data = base64.b64encode(f.read()).decode("utf-8")

    # 설계도에 결과를 끼워 넣어 손님에게 응답
    return render_template_string(PAGE, result=result, prob=prob, img_data=img_data)

# 이 파일을 직접 실행했을 때만 서버 시작
if __name__ == "__main__":
    app.run(debug=True)  # debug=True: 코드 고치면 자동 재시작 (개발용)