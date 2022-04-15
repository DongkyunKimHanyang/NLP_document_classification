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

### 방법
#### Question query 생성
저자는 Question에 대한 Answer Span을 찾는 방식으로 NER을 해결하고자 한다. 이때 Question을 일일히 만들어줘야 하는데,  
저자는 Question을 만드는 7가지 방법을 소개하고, 실험했다.  

-Position index of labels: label 번호를 Query으로 사용. ex) "one", "two", "three"  

-Keyword: label을 키워드로 변환 ex) ORG를 찾고자 할때는 -> "organization"을 query로 넣어준다.  

-Rule-based template filling: 간단한 템플릿을 사용해 label을 묻는 query를 생성한다. ex) "which organization is mentioned in the text". 

-Wikipedia: label의 위키피디아 정의를 query로 사용. ex) "an organization is an entity comprising multiple people ..."  

-Synonyms: label의 유의 단어를 Oxford Dictionary에서 찾아 query로 사용. ex) ORG를 찾을 때, "association"을 query로 사용.  

-Keyword + Synonyms: 말그대로 keyword와 synonyms를 이어붙여서 query로 사용. 

-Annotation guideline notes: label에 대한 가이드라인을 저자가 직접 작문하여 Query로 사용. ex) "find organizations including conpanies, agencies and institutions". 

#### Model details
Backbone은 [BERT](https://arxiv.org/abs/1810.04805) 모델을 사용했다.  
BERT에 <img width="474" alt="image" src="https://user-images.githubusercontent.com/87703352/163526570-0daad175-0d56-40ae-9693-0f3108f2c656.png"> 형태의 데이터를 입력, 여기서 q는 query의 토큰 그리고 x는 Context의 토큰들이다.  
그러면 BERT로부터 다음과 같은 형태의 표현형 벡터를 추출 할 수 있다. <img width="118" alt="image" src="https://user-images.githubusercontent.com/87703352/163526767-ed918248-81e7-40ed-87d8-2eee24fa1dba.png"> 여기서 n은 Context 토큰의 길이, d는 Hidden dimension.  
  
저자는 3개의 분류기 Head를 사용했다.  
1. Start Index binary classifier<img width="460" alt="image" src="https://user-images.githubusercontent.com/87703352/163528308-918a0cc8-acd7-4ac8-b985-5a6e1a153a69.png">
  
2. End Index binary classifier. End는 위와 동일  
3. Start-End matching classifier <img width="488" alt="image" src="https://user-images.githubusercontent.com/87703352/163529715-1601088e-2794-4a04-9415-365bf76d0e2f.png">
  

1.과 2.는 각 토큰이 Answer span의 Start/End Index인지 아닌지 이진 분류한다.  
예측된 Start-End matching probability 를 3.으로 매칭한다. Cutoff 값은 논문에 따로 나와있지 않다.  



