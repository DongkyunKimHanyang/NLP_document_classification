## RoBERTa를 활용한 NER
<img src = "https://user-images.githubusercontent.com/87703352/160063979-36f0bea7-7f78-4fcd-b42c-39315ab2c238.png" width="300" height="300">.  
RoBERTa는 토큰 별 표현형(Token representation)을 출력합니다. 표현형 벡터를 Dense layer나 Conditional Random field에 통과시켜 토큰별 분류를 수행 할 수있습니다.  

## RoBERTa + Dense
n_epoch = 5  
max_lr = 3e-5  
wramup_rate = 0.2  
w_decay = 0.001  

loss = Cross Entropy  
optimizer = adamW


## KLUE-NER
model - klue/roberta-base  

|label|f1-score|
|---|---|
|PS|0.94|
|LC|0.84|
|OG|0.85|
|DT|0.88|
|TI|0.94|
|QT|0.95|
|Macro|0.90|
  
  
## CoNLL2003
model - roberta-base  
|label|f1-score|
|---|---|
|PER|0.92|
|ORG|0.87|
|LOC|0.92|
|MISC|0.71|
|Macro|0.87|
