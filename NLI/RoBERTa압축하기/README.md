## RoBERTa
[RoBERTa](https://arxiv.org/abs/1907.11692)는 [BERT](https://arxiv.org/abs/1810.04805)모델을 기반으로 사전 학습 가정에서 Hyperparameter를 조정하여 좀더 강건하고 일반화 성능이 좋아진 모델입니다.  
BERT와 마찬가지로 RoBERTa-base 모델은 임베딩 layer + 12 Transformer layer로 구성됩니다.  
RoBERTa의 학습파라미터 숫자는 123만개로 매우 큰 모델이라고 할 수 있습니다.  
  
  
매우 좋은 성능을 내는 훌륭한 모델이지만, 연산량이 부담이 될때가 있어요.  
Downstream task를 수행 할때도 이렇게 큰 모델이 필요할까요?  
이 저장소에서는 Roberta의 Transformer layer를 뒤에서 부터 하나씩 제거하면서 성능이 언제까지 유지되는지 실험해보겠습니다.  


## 실험
[KLUE-NLI](https://klue-benchmark.com/) 데이터 셋에 대해 RoBERTa를 얼마나 작게 하면서, 성능을 유지 할 수있나 실험해보았습니다.  

model: klue/roberta-base. 

lr: 3e-5. 

optimizer: AdamW. 

scheduler: Warmup_decay. 


## 중간레이어에서 task 수행하기
<img src="https://user-images.githubusercontent.com/87703352/159858352-9aff4bff-c892-47b2-99bd-b6c446c9b0de.png" width="500" height="400"> 
Roberta의 중간 n 번째 layer에 Dense Classifier를 달아서 학습한 뒤, 성능을 측정해 보았습니다.
