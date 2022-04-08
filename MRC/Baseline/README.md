# MRC - baseline
BERT 모델 + Start/End position classifier로 구성한 모델을 구성한 저장소입니다. [Google AI reasearch의 BERT](https://github.com/google-research/bert), 
[KLUE-baseline] (https://github.com/KLUE-benchmark/KLUE-baseline) 등 다양한 레퍼런스를 참고해 작성한 코드입니다.

# Dataset
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/), [KLUE-MRC](https://klue-benchmark.com/)

# Model
Hugging Face에 배포되어있는 모델들 사용  
SQuAD: bert-base-case  
KorQuAD, KLUE: klue/roberta-base  

# Evaluation

## SQuAD 2.0
|Type|EM|F1|
|---|---|---|
|HasAnswer|73.82|80.51|
|NoAnswer|67.41|67.41|
|Total|70.61|73.95|  

## KorQuAD 1.0
|Type|EM|F1|
|---|---|---|
|HasAnswer|86.37|91.36|

## KorQuAD 1.0
|Type|EM|F1|
|---|---|---|
|HasAnswer|65.89|69.36|
|NoAnswer|72.72|72.72|
|Total|68.04|70.42|  
