# Single_Shot_Multibox_Detector
---------------------------------
## mnist 숫자 데이터를 활용한 SSD multibox Detector
- mnist 숫자 데이터를 활용해서 숫자의 local 값과 숫자의 classification 값을 얻어 올 수 있습니다.

-----------------
## Data 1 : Single Digit data
- `dataprovider.py`의 dataprovider 사용
- 하나의 이미지에 하나의 mnist 숫자 데이터 있는 dataprovider

### training data
- single mnist digit training data
- 1 채널 이미지 아닌 3 채널 이미지
  - <img width="700" src="https://user-images.githubusercontent.com/71860179/140700793-506a8ad2-16ae-4138-a97d-e0e5252d00f5.png">
- positive defualt box 시각화
  - <img width="700" src="https://user-images.githubusercontent.com/71860179/148726176-7b855eb3-043f-46a5-92ef-4669ac0d8f51.png">
- positive ground truth 시각화
  - <img width="700" src="https://user-images.githubusercontent.com/71860179/148726474-209010c2-24df-45f0-b5c7-991f14a4b3ac.png">


### test data
- single mnist digit test data
- 학습된 모델의 예측값으로부터 원하는 local 값과 class 값을 얻어 올 수 있습니다.
-  모델 예측 결과
  - <img width="700" src="https://user-images.githubusercontent.com/71860179/148726542-07bdb9e5-04d4-4191-90ca-de9d326b1bb8.png">
-----------------
## Data 2 : Mulitiple Digit data
- `multi_dataprovider.py`의 dataprovider 사용
- 하나의 이미지에 여러개(1개 ~ 3개)의 mnist 숫자 데이터 있는 dataprovider

### training data
- muliple mnist digit traing data
- 1 채널 이미지가 아닌 3채널의 이미지
  - <img width="700" alt="스크린샷 2022-01-12 오후 4 42 43" src="https://user-images.githubusercontent.com/71860179/149084688-c1524098-f749-4057-8d84-ef7c07f5e13d.png">
- positive default box 시각화
  -  <img width="700" alt="스크린샷 2022-01-12 오후 4 46 01" src="https://user-images.githubusercontent.com/71860179/149085141-ba450eaf-f886-4dcc-9856-f6f08ba05b6d.png">
- positive ground box 시각화 
  - <img width="700" alt="스크린샷 2022-01-12 오후 4 44 35" src="https://user-images.githubusercontent.com/71860179/149084953-cc3f4cbf-4efd-47a1-b57c-e0aa514697b4.png">

### test data
- multiple mnist digit test data
- 학습된 모델의 예측값으로부터 원하는 local 값과 class 값을 얻어 올 수 있습니다.
- 모델 예측 결과
  - <img width="700" alt="스크린샷 2022-01-12 오후 5 15 11" src="https://user-images.githubusercontent.com/71860179/149089229-35c34a06-cd6f-4137-b424-27bd13ad67e2.png">

