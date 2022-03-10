# Dataset
각각의 데이터셋

# 모델 구조
RoBERTa 모델을 인코더로 활용하고, Head로는 간단 Dense layer들을 사용한 모델을 구성했습니다.  
KLUE_NLI 데이터셋의 Backbone으로는 klue/roberta-large를,
SNLI 데이터셋의 Backbone으로는 roberta-large를 사용했습니다.

# 최적화
5 fold 앙상블 방법을 활용했습니다.  

<img src="https://user-images.githubusercontent.com/87703352/157607999-6d378763-2011-4672-bb45-178595d28a54.png" width="700" height="500">  
사용한 Hyperparameter는 아래와 같습니다  
Optimizer = [AdamW](https://arxiv.org/abs/1711.05101)  
Scheduler = Warmup  
learning_rate = 3e-5  
warmup_rate = 0.3

# 5Fold의 모델별 validset 정확도
|Model|KLUE_NLI|SNLI|
|---|---|---|
|1|91.68%||
|2|90.80%||
|3|91.08%||
|4|91.40%||
|5|90.64%||

# Test set에 대한 정확도
|Model|KLUE_NLI|SNLI|
|---|---|---|
|1|89.90%||
|2|90.40%||
|3|89.63%||
|4|90.33%||
|5|89.47%||
|SoftVoting|91.37%|||
