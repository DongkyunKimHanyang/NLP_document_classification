# NLP_document_classification
Huggingface &amp; Pytorch로 문서 분류 모델 생성 예제 코드입니다.  
RoBERTa-base 모델을 Backbone으로 사용하고, 기본 적인 layer들을 연결해서 문서를 분류해보도록 하겠습니다.  
[RoBERTa](https://arxiv.org/abs/1907.11692) 모델과 그 근간인 [BERT](https://arxiv.org/abs/1810.04805) 모델의 자세한 사항은 링크를 타고 들어가 논문을 읽어보시길 추천드립니다.  

# Dataset - IMDB 리뷰 데이터
IMDB는 이진 감성 분류를 위한 영화 리뷰 데이터 입니다. 
Label은 긍정(1) 또는 부정(0)으로 나뉘어 지며,  
25,000개의 training set, 25,000개의 test set 샘플들을 포함하고 있습니다.  
해당 코드 메인문을 실행하면 자동으로 HuggingFace의 Dataset Library로 IMDB 데이터를 불러와 사용합니다.  
  
데이터 예시
|Text|Label|
|---|---|
|I can't believe that those praising this movie herein aren't thinking of some other film...|0(Negative)|
  
# 모델 구성
Backbone 모델로은 HuggingFace에서 RoBERTa-base 모델을 가져와 사용했습니다.
Fine-tuning은 아래 3가지 방식으로 해볼게요

1. [CLS] token pooling + Dense:  
RoBERTa의 last_hidden_state에서 첫번째 토큰의 hidden vector를 Dense에 입력해서 분류를 수행하는 방식입니다.

<img src="https://user-images.githubusercontent.com/87703352/156522668-beaf45da-b150-4af8-b5da-3ea5fad4eaab.png" width="400" height="500">

2. Weighted average pooling + Dense:  
RoBERTa의 last_hidden_state에서 모든 토큰 벡터를 weighted average pooling으로 1차원으로 만들어 Dense에 입력하는 방식입니다.

<img src="https://user-images.githubusercontent.com/87703352/156523926-ead773aa-add5-4cb1-8bc9-ccf92c02c957.png" width="700" height="500">

3. LSTM + Dense:  
RoBERTa의 last_hidden_state를 LSTM layer + Dense layer에 입력해서 분류를 수행
<img src="https://user-images.githubusercontent.com/87703352/156526848-8a980d7a-3616-4dcd-8c6f-3825adda8a55.png" width="700" height="500">

# Version
Ubuntu == 20.04  
python == 3.8.10  
torch == 1.10.2  
CUDA == 11.3  

# 실행
아래 처럼, _3_main.py 파일에 argment들을 입력하여 실행 시키면됩니다.  
  
python3 _3_main.py --head_name [cls or weight_avg or lstm] -- train_batch_size [int] --test_batch_size [int] --lr [float] --warup_rate [float] --total_epochs [int]  
  
--**head_name**는 cls, weight_avg, lstm은 각각 1,2,3번 방식에 해당하는 헤드를 고르는것 입니다.  
--**train_batch_size**는 학습과정에서의 배치 크기를 결정합니다.  
--**test_batch_size**는 추론과정에서의 배치 크기를 결정합니다.  
--**lr**은 warmup scheduler 방식에서 최고점 lr을 뜻합니다.  
--**warmup_rate**는 전체 step중 warmup이 차지하는 비율입니다.  
--**total_epochs**는 총 학습 epoch를 결정합니다.  
예시:  
```
python3 _3_main.py --head_name dense -- train_batch_size 16 --test_batch_size 256 --lr 3e-5 --warup_rate 0.3 --total_epochs 3
```

# 결과
train_batch_size = 16, lr =3e-5, warmup_rate=0.3, total_epochs=3으로 했을때 epoch별 test_set 정확도입니다.  
|Epoch|[CLS] token pooling + Dense|Weighted average pooling + Dense|LSTM + Dense|
|---|---|---|---|
|1|93.41|93.94|93.74|
|2|95.32|95.33|94.67|
|3|95.60|95.42|95.52|

# 결과
단순히 Roberta-base 모델을 Fine tuning 하는것 만으로도 95%이상의 성능을 얻을 수 있었습니다.  
하지만 1,2,3번 방식
