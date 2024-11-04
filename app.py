from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import os
import numpy as np
# import bentoml

app = Flask(__name__)

BENTO_API_URL = "https://a-yo-image-fbf92702.mt-guc1.bentoml.ai/generate_frames"

# 업로드 설정
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    """
    이미지 파일 경로를 받아서 전처리 후 모델 입력 형식으로 변환하는 함수입니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        repeat_count (int): 이미지를 반복하는 횟수 (디폴트는 10)
        
    Returns:
        dict: 전처리된 이미지와 라벨을 포함한 딕셔너리
    """
    # 이미지 로드 및 전처리
    img = Image.open(image_path)
    img_array = np.array(img)

    # JSON 형식으로 반환
    return {"input_array": img_array.tolist()}


# 확장자 체크 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 루트 라우트
@app.route('/')
def index():
    return render_template('upload.html')

# 파일 업로드 라우트
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = "test_image.png"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('display_file', filename=filename))
    return redirect(request.url)

# 업로드된 이미지 표시 라우트
@app.route('/display/<filename>')
def display_file(filename):
    return render_template('upload.html', filename=filename)

# POST 요청: 이미지 전처리 후 BentoML에 전송
@app.route('/motion', methods=['POST'])
def create_motion_post():
    # 이미지 파일 경로 설정
    image_path = "static/uploads/test_image.png"
    
    # 이미지 전처리 수행
    payload = preprocess_image(image_path)
    
    # POST로 데이터 전송
    response = requests.post(BENTO_API_URL, json=payload)
     # 응답 상태 코드 및 내용 출력
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    # 예측 결과 확인
    if response.status_code == 200:
        result = response.json()
        return f"모션이 생성되었습니다! 결과: {result}"
    else:
        return "모션 생성에 실패했습니다."

if __name__ == '__main__':
    app.run(debug=True)
