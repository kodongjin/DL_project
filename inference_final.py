import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import os
from PIL import Image

# 모델 로드
model = tf.keras.models.load_model(r'C:\Users\sinha\Deep_Learning\result_model.h5')

# 클래스명 리스트
class_names = ['강아지 짖는 소리', '고양이 우는 소리', '발걸음 소리', '망치질 소리', '문 여닫는 소리', '야외 놀이터 소리', '사이렌 소리', '아이 울음소리']

# 오디오 녹음
def record_audio(duration=10, sample_rate=44100):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data

# 기존 오디오 마지막 5초를 가져와서 새로운 오디오 5초 추가
def add_audio(existing_audio, duration=5, sample_rate=44100):
    new_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.concatenate((existing_audio[-5*sample_rate:], new_audio))

# 전역 변수로 저장된 스펙트로그램 이미지 파일 경로 초기화
full_path = ""

def plot_spectrogram(audio_data, sample_rate=44100, save_path=r'C:\Users\sinha\Deep_Learning\spectrogram_file'):
    global full_path  # 전역 변수 사용 선언
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = librosa.feature.melspectrogram(y=np.squeeze(audio_data), sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_position([0, 0, 1, 1])
    ax.set_ylim(0, 8000)
    ax.axis("off")
    img = librosa.display.specshow(S_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)

    S_dB[:, 0:10] = 0 #결측치 처리 부분
    S_dB[:, 500:510] = 0

    # 저장 파일 확인
    os.makedirs(save_path, exist_ok=True)
    filename = 'latest_spectrogram.png'  # 고정된 파일명 사용(중복으로 처리하여 스펙트로그램 파일 여러게 생기는거 방지)
    full_path = os.path.join(save_path, filename)
    
    plt.savefig(full_path)
    plt.close(fig)
    return S_dB

def predict_class(image_path, model):
    # 이미지 파일을 RGB 형식으로 불러오기
    image = Image.open(image_path).convert('RGB')  # A채널 삭제
    
    # 모델의 예상 입력 크기 확인
    input_shape = model.input_shape[1:3]  # (height, width) 형태
    
    # 이미지를 모델 입력 크기에 맞게 변환
    image = image.resize(input_shape[::-1])  # width, height 순서대로 resize
    
    # 이미지를 배열로 변환하고, 배치 차원 추가
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # 배치(None) 차원 추가
    
    # 모델 예측 수행
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    max_prediction = np.max(predictions)
    
    # 확률 값이 0.5보다 낮은 경우 이벤트 소음 없음 반환
    if max_prediction < 0.5:
        return '이벤트 소음 없음'
    else:
        return class_names[predicted_class_index]

#웹 구현 및 실행
#초기 설정
sample_rate = 44100
initial_duration = 10
additional_duration = 5

st.set_page_config(layout="wide")

# 스타일 설정
st.markdown("""
<style>
.centered {
    text-align: center;
}
table-container {
    width: 100%;
}
table {
    width: 80%;
    margin: auto;
}
</style>
<div class='centered'>
    <h1>이벤트 인식을 활용한</h1>
    <h1>IoT 기반 가정 안전 솔루션 개발</h1>
</div>
""", unsafe_allow_html=True)

# 우측 상단에 텍스트 추가
st.markdown("<h3 style='text-align: right;'>2024 Deep Learning Team10</h3>", unsafe_allow_html=True)

button_placeholder = st.empty()  # 버튼을 동적으로 제어할 placeholder

if button_placeholder.button("녹음 시작", key='1'):
    button_placeholder.empty()  # "녹음 시작" 버튼 제거
    st.markdown("<h2 class='centered'>인식 중...</h2>", unsafe_allow_html=True)
    prediction_placeholder = st.empty()  # 예측 결과를 갱신할 placeholder 생성
    time_placeholder = st.empty()  # 업데이트 시각을 표시할 placeholder 생성
    
    audio_data = record_audio(initial_duration, sample_rate)
    plot_spectrogram(audio_data, sample_rate)
    predicted_class = predict_class(full_path, model)
    
    # 초기 예측 결과 표시
    with prediction_placeholder.container():
        st.markdown(f"""
        <div class="table-container">
            <table>
                <tr><th>예측한 클래스</th></tr>
                <tr><td><h1 style='color: red; text-align: center;'>{predicted_class}</h1></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # 업데이트 시각 표시
    with time_placeholder.container():
        st.markdown(f"""
        <div class="table-container">
            <table>
                <tr><th>업데이트 시각</th></tr>
                <tr><td><h2 style='text-align: center;'>{datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}</h2></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # 5초 추가 녹음 및 결과 업데이트 반복
    while True:
        audio_data = add_audio(audio_data, additional_duration, sample_rate)
        plot_spectrogram(audio_data, sample_rate)
        predicted_class = predict_class(full_path, model)
        
        # 결과 갱신
        with prediction_placeholder.container():
            st.markdown(f"""
            <div class="table-container">
                <table>
                    <tr><th>예측한 클래스</th></tr>
                    <tr><td><h1 style='color: red; text-align: center;'>{predicted_class}</h1></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # 시각 갱신
        with time_placeholder.container():
            st.markdown(f"""
            <div class="table-container">
                <table>
                    <tr><th>업데이트 시각</th></tr>
                    <tr><td><h2 style='text-align: center;'>{datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}</h2></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
