# NLP_document_classification
Huggingface &amp; Pytorch로 문서 분류 모델 생성 예제 코드입니다.  
RoBERTa-base 모델을 Backbone으로 사용하고, 기본 적인 Dense layer들을 연결해서 문서를 분류해보도록 하겠습니다.  
[RoBERTa](https://arxiv.org/abs/1907.11692) 모델과 [BERT](https://arxiv.org/abs/1810.04805) 모델의 자세한 사항은 링크를 타고 들어가 논문을 읽어보시길 추천드립니다.  

# Dataset - IMDB 리뷰 데이터
IMDB는 이진 감성 분류를 위한 영화 리뷰 데이터 입니다. 
Label은 긍정(1) 또는 부정(1)으로 나뉘어 지며,  
25,000개의 training set, 25,000개의 test set 샘플들을 포함하고 있습니다.  
해당 코드 메인문을 실행하면 자동으로 HuggingFace의 Dataset Library에서 다운로드되어 사용됩니다.

# 모델 구성
Backbone 모델로은 HuggingFace에서 RoBERTa-base 모델을 가져와 사용했습니다.
Fine-tuning은 아래 3가지 방식으로 해볼게요

1. [CLS] token pooling + Dense
RoBERTa의 last_hidden_state에서 첫번째 토큰의 hidden vector를 Dense에 입력해서 분류를 수행하는 방식입니다.

<img src="https://user-images.githubusercontent.com/87703352/156522668-beaf45da-b150-4af8-b5da-3ea5fad4eaab.png" width="400" height="500">

2. Weighted average pooling + Dense
RoBERTa의 last_hidden_state에서 모든 토큰 벡터를 weighted average pooling으로 1차원으로 만들어 Dense에 입력하는 방식입니다.

<img src="https://user-images.githubusercontent.com/87703352/156523926-ead773aa-add5-4cb1-8bc9-ccf92c02c957.png" width="700" height="500">

3. LSTM + Dense
RoBERTa의 last_hidden_state를 LSTM layer + Dense layer에 입력해서 분류를 수행
<img src="https://user-images.githubusercontent.com/87703352/156526848-8a980d7a-3616-4dcd-8c6f-3825adda8a55.png" width="700" height="500">

#Version
Ubuntu == 20.04
python == 3.8.10
torch == 1.10.2
CUDA == 11.3

#실행
python3 _3_main.py --head_name [dense or avg_dense or lstm] -- train_batch_size [int] --test_batch_size [int] --lr [float] --warup_rate [float] --total_epochs [int]  
예시:
```
python3 _3_main.py --head_name dense -- train_batch_size 16 --test_batch_size 256 --lr 3e-5 --warup_rate 0.3 --total_epochs 10
```
