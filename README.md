# Loan_Forecasting
MNC에서 주관한 대출자 채무 불이행 예측 모델 구현 대회입니다.<br/><br/>

## **대출자 채무 불이행 예측 모델 구현**

- Tree 기반 모델들을 사용하였으며, 점수를 올리기 위해 Hard voting을 하였습니다.
- train data를 5개로 분류하여, 모델별로 4개씩 교차로 학습을 시키는 전략을 사용하였습니다.<br/><br/>

## 대회 개요
- **대회 기간** : 2021년 1월 26일 09:00 AM ~ 2021년 2월 8일 12:00 PM
- **문제 정의** : p2p 대부업체의 고객 데이터를 활용한 채무 불이행 여부 예측 과제 수행
- **추진 배경**
	-  핀테크 분야 성장에 따른 금융 분야 AI 활용도 증가
	- 금융 서비스 고객의 분류를 통한 금융 시장 건전성 제고 목표


- **평가 지표** 
	- **Macro F1 Score**
		- F1 Score는 정밀도(precision)와 재현율(recall)을 모두 고려하기 위해 두 가지를 조화 평균을 통해 계산하는 지표
	![image](https://github.com/NOTITLEUNTITLE/Loan_Forecasting/blob/main/image.PNG?raw=true)
<br/><br/><br/>

## 문제 접근 방법
Tree기반 모델을 사용하였으며, 10만개의 train data에 대하여 부분적인 학습을 진행하였습니다.<br/>
부분적인 학습이란, 10만개를 5등분하여 2만개씩 5개로 나눕니다.<br/>
그다음 Tree기반 모델을 5개 만들고, 8만개의 데이터를 각각 학습시켜줍니다.<br/>
```python
model1 = [train_data1, train_data2, train_data3, train_data4] # 5번 제외
model2 = [train_data1, train_data2, train_data3, train_data5] # 4번 제외
model3 = [train_data1, train_data2, train_data4, train_data5] # 3번 제외
model4 = [train_data1, train_data3, train_data4, train_data5] # 2번 제외
model5 = [train_data2, train_data3, train_data4, train_data5] # 1번 제외
```
이런식으로 학습을 진행한 후, 모델별로 나온 prediction 값을 hard voting을 하였습니다.<br/>
model1이 학습하지 않은 5번 데이터셋에 대해서는 2,3,4,5 모델들이 학습을 진행해주었으므로 더 많은 데이터셋을 학습시키는 효과를 얻을수 있었습니다.<br/>
끝으로 threshold 값을 평가지표인 f1 score에 맞추어,<br/>
최댓값을 구해주는 함수를 작성해준뒤, 최적화 작업을 진행하였습니다.<br/>

```python
# threshold값을 최적화 시키는 함수.
def calc_score_model(model, name, X_train, y_train, X_val, y_val):
    model1 = model
    model1.fit(X_train, y_train)

    y_pred1 = model1.predict(X_val)
    y_prob1 = model1.predict_proba(X_val)

    thr_result = 0.5
    max_val = 0.0

    scale = 1000
    # 반목문을 통하여 threshold값을 0.5 ~ 0.0005까지 조정하면서 calc_sum_f1_and_accuracy()값이 제일 높은 threshold값을 찾습니다.
    for thr in range(1, scale):
        val = calc_sum_f1_and_accuracy(y_val, (y_prob1[:,1] >= thr / scale))
        if val > max_val:
            thr_result = thr / scale
            max_val = val
    
    return [name, thr_result, max_val, model]
```
<br/>

- XGBClassifier 모델 5개를 사용해도 꽤 좋은 점수를 얻을수 있었습니다.
<br/><br/><br/><br/>


## 학습 결과
```
Cat 0.687737265883217 0.363
LGBM 0.68583139448173 0.368
Gradient 0.6675123797244247 0.366
rnd 0.6484557297215036 0.381
Decision 0.5765703389830508 0.001
-------------
Cat 0.6897985839740425 0.36
LGBM 0.6868351554439782 0.342
Gradient 0.6698568019093079 0.366
rnd 0.6574241057158079 0.391
Decision 0.578490976771888 0.001
-------------
LGBM 0.6903328657056704 0.372
Cat 0.6884725601004664 0.357
Gradient 0.6671463209530484 0.335
rnd 0.6550289702233251 0.381
Decision 0.579489700770742 0.001
-------------
Cat 0.6875170129689026 0.333
LGBM 0.6852640684410647 0.391
Gradient 0.6638816281441196 0.367
rnd 0.6566285560913132 0.381
Decision 0.5780770874675623 0.001
-------------
Cat 0.6864923592493297 0.31
LGBM 0.6862856898517673 0.345
Gradient 0.668553090092721 0.349
rnd 0.6535108073505047 0.381
Decision 0.5826743902903742 0.001
```

## 마무리
- 저는 hyper-parameter tuning을 하지 않았으며, optuna나 혹은 random-search, grid-search 등도 하지 않았습니다.
- 물론 처음에는 hyper-parameter tuning을 열심히 하였으나, 대회종료가 다가올수록 점수 올리기에 혈안이 되어 있었습니다...ㅜㅜ
- 각 모델별로 튜닝을 진행하거나 혹은 XGBClassifier 모델만 사용해서 튜닝을 한후, hard voting을 진행하는 것도 괜찮아 보였습니다!


<br/><br/><br/>
	
-------
### Loan_prediction.ipynb : 최종 제출파일
### pr_report.html : sklearn의 라이브러리를 활용한 EDA Viewer
