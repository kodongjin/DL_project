# 음향 이벤트 인식을 활용한 IoT 기반 가정 안전 솔루션 개발
본 repository는 2024년 서울과학기술대학교 산업공학과 딥러닝 수업에서, 10팀이 수행한 프로젝트를 재현할 수 있도록 구성되었습니다.
구체적인 진행 내용은 아래의 발표자료를 참고하시기 바랍니다.

(발표자료)

팀원 구성은 다음과 같습니다.
+ 고동진 (smileboy816@naver.com)
  +  데이터 수집 및 합성:
      + 배경음악과 이벤트 음성 데이터를 구성 및 합성하였습니다.
      + 실제 상황을 반영한 랜덤함수를 활용하여 자연스러운 데이터 합성하였습니다. 
  +  데이터 전처리:
      + 데이터 불균등 문제 해결을 위해 언더샘플링 적용하여 각 클래스당 400개의 데이터로 설정하여 불균등을 처리하였습니다. 
      + 다양한 경우의 수를 고려하여 자연스러운 데이터 합성 시도하였습니다.
      + 배경소음과 이벤트 데이터의 길이를 10초로 조정하여 학습데이터 만들었습니다. 
      + 샘플레이트 기준 조사 및 리샘플링 수행하였습니다. 
  +  음성 데이터 주파수 축 분석:
      + 멜스펙트로그램 생성 및 푸리에 변환을 활용하여 음성 데이터를 이미지로 분석할 수 있도록 준비하였습니다.
      + 데이터 전처리 과정 수행 및 추가 이벤트 데이터 전처리 지원하였습니다.
      
+ 김도영 (kimdoyoung1023@seoultech.ac.kr)
  
  + 모델 학습
    + 모델 학습과 관련된 모든 과정을 담당하여 수행하였습니다.
    + 서버에 데이터셋을 확보하여, 전처리 및 합성 과정을 수행하였습니다.
    + GPU를 활용하여 총 10가지 서로 다른 실험을 시행하였고, 결과를 정리하였습니다.
    
+ 곽동욱 (charles2659@naver.com)
  
# 0. 데이터셋 다운로드
본 프로젝트에서는 배경소음과 이벤트 음향을 합성하여, 실내 상황에서 들을 수 있는 음향 데이터셋을 구축하고 이것을 딥러닝 모델 학습 및 평가에 사용합니다.
데이터의 출처와 활용 내용은 아래와 같습니다.

| **데이터** |**출처** | **활용내용** |
|------|--------|--------|
| 드럼세탁기소리 | [생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296)  | 배경소음 |
| 통돌이세탁기소리 | [생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296)  | 배경소음 |
| 진공청소기소리 | [생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296)  | 배경소음 |
| 식기세척기소리 | [생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296)  | 배경소음 |
| 음악흥얼거리는소리 |[배경소음 데이터셋 - 소리의 비언어적인 부분까지 감지할 수 있는 인공지능 데이터셋](https://selectstar-opendataset2021.s3.ap-northeast-2.amazonaws.com/%EC%BD%94%ED%81%B4%EB%A6%AC%EC%96%B4%EB%8B%B7/%EC%9D%8C%EC%95%85+%ED%9D%A5%EC%96%BC%EA%B1%B0%EB%A6%AC%EB%8A%94+%EC%86%8C%EB%A6%AC.zip) | 배경소음 |
| 화장실소리 | [배경소음 데이터셋 - 소리의 비언어적인 부분까지 감지할 수 있는 인공지능 데이터셋](https://selectstar-opendataset2021.s3.ap-northeast-2.amazonaws.com/%EC%BD%94%ED%81%B4%EB%A6%AC%EC%96%B4%EB%8B%B7/%ED%99%94%EC%9E%A5%EC%8B%A4.zip) | 배경소음 |
| 부엌소리 | [배경소음 데이터셋 - 소리의 비언어적인 부분까지 감지할 수 있는 인공지능 데이터셋](https://selectstar-opendataset2021.s3.ap-northeast-2.amazonaws.com/%EC%BD%94%ED%81%B4%EB%A6%AC%EC%96%B4%EB%8B%B7/%EB%B6%80%EC%97%8C.zip) | 배경소음 |

| **데이터** |**출처** | **활용내용** |
|------|--------|--------|
| 강아지짖는소리 | [도시 소리 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=585) | 이벤트소음 |
| 고양이우는소리 | [도시 소리 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=585)  | 이벤트소음 |
| 발걸음소리 | [도시 소리 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=585)  | 이벤트소음 |
| 망치질소리 | [생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296) | 이벤트소음 |
| 문여닫는소리 |[생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71296) | 이벤트소음 |
| 야외놀이터소리 | [URBANSOUND8K DATASET](https://urbansounddataset.weebly.com/urbansound8k.html)| 이벤트소음 |
| 사이렌 | [URBANSOUND8K DATASET](https://urbansounddataset.weebly.com/urbansound8k.html) | 이벤트소음 |
| 아이울음소리| [Infant cry Dataset](https://www.kaggle.com/datasets/sanmithasadhish/infant-cry-dataset) | 이벤트소음 |

다운로드 받은 데이터셋을 아래와 같은 구조로 정리하십시오.

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

# 1. 데이터셋 전처리
데이터셋에 대한 전처리는 크게 '파일명 변경'과 '데이터 합성'으로 이루어집니다.
데이터셋 전처리를 위해, 아래 코드를 차례대로 실행하십시오.


'파일명 변경'을 통해 **2young_14415140_드럼세탁기 소리.wav -> 드럼세탁기소리_0001.wav**와 같이 파일의 이름을 변경할 수 있습니다.

    python ch_filename.py
   
'데이터 합성'을 통해 배경소음과 이벤트 음향을 합성할 수 있습니다. 또한 합성한 음성에 대해 wav 파일과 Mel Spectrogram 이미지로 저장할 수 있습니다.

최 과정에서는 Inference 과정에서의 데이터 결측치를 감안하여 멜스펙트로그램에서 나타내는 주파수 대역을 줄이고, 0초와 5초 부근에 0.1초씩 패딩을 추가로 진행하였습니다. 해당 처리를 원하지 않는다면, **data_systhesis_original.py**를 실행하십시오.

    python data_systhesis.py

+  **Mel Spectrogram made by data_systhesis_original.py**
  ![image](https://github.com/kodongjin/DL_project/assets/133321474/859cd5c4-ffe7-474c-ad1e-5b798da8a3f1)

+ **Mel Spectrogram made by data_systhesis.py**
![image](https://github.com/kodongjin/DL_project/assets/133321474/130a2085-b958-4ebe-9dc9-554b15e6b091)

# 2. 모델 학습
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
|Mel_CNN(Generalization-ReduceLR)|59.76 MB|0.7220|
|Mel_CNN(Generalization-Dropout,ReduceLR)|59.76 MB|**0.7422**|

가장 높은 성능을 낸 **Mel_CNN(Generalization-Dropout,ReduceLR)** 구조에 대하여 학습 코드를 제공합니다.

    python train.py

추가적으로, 학습된 모델을 제공합니다.

+ [A trained model using a dataset processed by data_systhesis_original.py](https://drive.google.com/file/d/1Snaj06LRUnug9wvUan047H8OR-9z4Ghq/view?usp=sharing) (Val_Acc : 0.7422)
+ [A trained model using a dataset processed by data_systhesis.py](https://drive.google.com/file/d/1vSqKABHLLtc3SgJUGimdWoKj7k2TRJMe/view?usp=sharing) (Val_Acc : 0.7303)

# 3. 모델 평가 및 시각화
학습한 모델로 test set에 대한 Classification을 진행하고, 분류 결과를 시각화합니다.

**data_systhesis_original.py**로 데이터 및 모델을 생성했다고 가정합니다.

    python test.py

**아래와 같은 형식의 Confusion Matrix를 얻을 수 있습니다.**
![image](https://github.com/kodongjin/DL_project/assets/133321474/0d8045e4-f2ce-4307-abba-bac65b2b8c6c)

# 4. Inference
녹음 장치가 있는 디바이스(ex : 노트북)에서 웹(streamlit) 기반으로 작동하는 인퍼런스 파일입니다.
녹음된 음성은 후처리 과정(발표자료 참고)을 거쳐, 모델의 입력으로 들어갑니다.

    pip install streamlit
    streamlit run inference.py

**작동 예시**


https://github.com/kodongjin/DL_project/assets/133321474/0e6289a6-ef8d-46fa-9122-03aae6683001







