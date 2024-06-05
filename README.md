![image](https://github.com/kodongjin/DL_project/assets/133321474/bb967a3e-0f9b-49e4-ac13-d849a5b71cd7)# 음향 이벤트 인식을 활용한 IoT 기반 가정 안전 솔루션 개발
본 repository는 2024년 서울과학기술대학교 산업공학과 딥러닝 수업 프로젝트 10팀이 수행한 작업을 재현할 수 있도록 구성되었습니다.
구체적인 진행 내용은 아래의 발표자료를 참고하기 바랍니다.
(발표자료)

팀원 구성은 다음과 같습니다.
+ 고동진 (메일주소)
+ 김도영 (kimdoyoung1023@seoultech.ac.kr)
+ 곽동욱 (메일주소)
  
# 0. 데이터셋 다운로드 및 전처리
## a. 데이터셋 다운로드
본 프로젝트에서는 배경소음과 이벤트 음향을 합성하여, 실내 상황에서 들을 수 있는 음향 데이터셋을 구축하고 이것을 딥러닝 모델 학습 및 평가에 사용합니다.
데이터의 출처와 활용 내용은 아래와 같습니다.

| **데이터** |**출처** | **활용내용** |
|------|--------|--------|
| 드럼세탁기소리 | [생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296)  |배경소음|
| 1 |1   |2|
| 13 | 1   |3|
| 13 | 1   |3|

다운로드 받은 데이터세트를 아래와 같은 구조로 정리하십시오.

    Data/
    ├── Background/
    │   ├── 드럼세탁기소리/
    │   │   ├── 2young_14415140_드럼세탁기 소리.wav
    │   │   ├── jjuu1230_14348875_드럼세탁기 소리.wav
    │   │   └── ...
    │   ├── 부엌소리/
    │   │   ├── ...
    │   └── ...
    ├── Event/
    │   ├── 강아지짓는소리/
    │   │   ├── ...
    │   ├── 고양이우는소리/
    │   │   ├── ...
    │   └── ...

## b. 데이터셋 전처리
데이터셋에 대한 전처리는 크게 '파일명 변경'과 '데이터 합성'으로 이루어집니다.
데이터셋 전처리를 위해, 아래 코드를 차례대로 실행하십시오.


1) '파일명 변경'을 통해 **2young_14415140_드럼세탁기 소리.wav -> 드럼세탁기소리_0001.wav**와 같이 파일의 이름을 변경할 수 있습니다.

    python ch_filename.py
   

3) '데이터 합성'을 통해 배경소음과 이벤트 음향을 합성할 수 있습니다. 또한 합성한 음성에 대해 wav 파일과 Mel Spectrogram 이미지로 저장할 수 있습니다.

본 과정에서는 Inference 과정에서의 데이터 결측치를 감안하여 멜스펙트로그램에서 나타내는 주파수 대역을 줄이고, 0초와 5초 부근에 0.1초씩 패딩을 추가로 진행하였습니다. 해당 처리를 원하지 않는다면, **data_systhesis_original.py**를 실행하십시오.

    python data_systhesis.py


## 1. 모델 학습
학습 및 평가는 **data_systhesis_original.py**로 생성한 데이터에 대해 진행되었습니다.

모델의 입력 형식과 구조에 따라 다양한 실험을 진행하였으며, 그 결과는 다음과 같습니다.

성능은 검증 데이터셋의 8개 클래스에 대한 분류 정확도를 나타냅니다.


|실험|파라미터|성능|
|------|---|---|
|Wav_MLP|39.23 MB|0.2271|
|Wav_Conv1D|55.47 MB|0.1240|
|Mel_MLP|49.51 MB|0.4929|
|Mel_CNN|59.76 MB|0.7129|
|Mel_CNN(Imagenet)|59.76 MB|0.4976|
|Mel_CNN(Generalization-Dropout)|59.76 MB|0.7309|
|Mel_CNN(Generalization-BatchNorm)|59.78 MB|0.6857|
|Mel_CNN(Generalization-ReduceLR)|59.76 MB||
|Mel_CNN(Generalization-Dropout,ReduceLR)|59.76 MB||

가장 높은 성능을 낸 **Mel_CNN(Generalization-Dropout,ReduceLR)** 구조에 대하여 학습 코드를 제공합니다.

    python train.py


