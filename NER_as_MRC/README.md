논문 : [Xiaoya Li et al. A Unified MRC Framework for Named Entity Recognition.](https://arxiv.org/abs/1910.11476)의 간단 요약과 코드를 정리해 둔 페이지입니다.  
위 논문의 공식 [Github 주소](https://github.com/ShannonAI/mrc-for-flat-nested-ner)  
여기서는 제 나름 대로 코드를 다시 정리해 보고, 한국어 KLUE-NER 데이터셋에서도 실험을 해보겠습니다.

## 논문 리뷰
### 연구배경  
Nested NER이란 아래 그림 처럼 Entity 안에 하위 Entity가 겹쳐서 들어있는것을 뜻한다.  

<img width="731" alt="image" src="https://user-images.githubusercontent.com/87703352/163115249-db98a853-07df-4fd5-b0ce-717a3ef13ae8.png"> 

NER을 해결하기 위해 Hierarchical LSTM+CRF 방식 ([Ju et al., 2018](https://aclanthology.org/N18-1131))  
BERT 기반 Anchor-Region-Network 방식 ([Lin et al., 2019](https://arxiv.org/abs/1906.03783))   
Dynamic span graph 방식 ([Luan et al., 2019](https://arxiv.org/abs/1904.03296))이 제안 되어 왔다.  

### 연구목적  
저자는 Nested/Flat NER을 모두 해결할 수 있는 Unified MRC for NER을 제안한다.  
제안 된 방법은 (Context, Label)을 포함하는 pair NER 데이터를 (Question, Answer, Context) triple로 변환하여 모델이 MRC task를 수행하게 한다.  

예시)  PER Entity 추출  
Context: "[Washington] was born into slavery on the farm of James Burroughs"  
Question: "Which person is mentioned in the text?"  
Answer: "Washington"
