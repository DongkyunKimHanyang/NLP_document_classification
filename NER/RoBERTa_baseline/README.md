## RoBERTa를 활용한 NER
<img src = "https://user-images.githubusercontent.com/87703352/160063979-36f0bea7-7f78-4fcd-b42c-39315ab2c238.png" width="300" height="300">.  
RoBERTa는 토큰 별 표현형(Token representation)을 출력합니다. 표현형 벡터를 Dense layer나 Conditional Random field에 통과시켜 토큰별 분류를 수행 할 수있습니다.  

## RoBERTa + Dense
먼저 간단하게, RoBERTa의 토큰 표현형을 Dense layer로 분류해보겠습니다.  
