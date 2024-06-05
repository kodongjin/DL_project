# 음향 이벤트 인식을 활용한 IoT 기반 가정 안전 솔루션 개발
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

| 모델 | 정확도 |
|------|--------|
| 모델 1 | 0.85   |
| 모델 2 | 0.90   |
| 모델 3 | 0.88   |

## 1. 연구 주제 및 선정 이유
본 연구는 IoT 기기를 통해 일상 소음에 섞인 특정 위험 상황의 소리를 탐지하는 것을 목표로 합니다. 이러한 탐지 시스템은 실제 생활 환경에서 유용하게 적용될 수 있으며, 스마트 홈 기기를 통한 사고 예방 및 대응 시간 단축에 기여할 수 있습니다. 스마트 홈 기기를 활용한 실시간 위험 감지 기술은 가정 내 안전을 강화하고 재난 발생 시 신속한 대응을 가능하게 합니다.

기존의 단순한 소리 인식을 넘어서 실제 생활 소음 속에서 위험 상황을 정확하게 구분할 수 있는 기술의 개발은 사회적 가치가 크며, 기술적으로도 큰 도전입니다.
