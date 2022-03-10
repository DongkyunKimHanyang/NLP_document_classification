# Dataset
각각의 데이터셋

# 모델 구조
RoBERTa 모델을 인코더로 활용하고, Head로는 간단 Dense layer들을 사용한 모델을 구성했습니다.  

# 최적화
5 fold 앙상블 방법을 활용했습니다.

사용한 Hyperparameter는 아래와 같습니다  
Optimizer = [AdamW](https://arxiv.org/abs/1711.05101)  
Scheduler = Warmup  
learning_rate = 3e-5  
warmup_rate = 0.3

#Fold별 validset 정확도

각 fold에 대한 모델 각각 서로 다른 최적 Hyperparameter를 찾고 추론시 Soft Voting으로 예측합니다.
  
