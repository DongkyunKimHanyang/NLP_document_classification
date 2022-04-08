# MRC - baseline
BERT 모델 + Start/End position classifier로 구성한 모델을 구성한 저장소입니다. [Google AI reasearch의 BERT](https://github.com/google-research/bert), 
[KLUE-baseline] (https://github.com/KLUE-benchmark/KLUE-baseline) 등 다양한 레퍼런스를 참고해 작성한 코드입니다.

# Dataset
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/), [KLUE-MRC](https://klue-benchmark.com/)

# Model
Hugging Face에 배포되어있는 모델들 사용  
SQuAD: bert-base-case  
KorQuAD, KLUE: klue/roberta-base  
