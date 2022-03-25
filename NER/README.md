# Named Entity Recognition - NER
개체명 인식 (NER)의 정의 -  
미리 정의해 둔 사람, 회사, 장소, 시간, 단위 등에 해당하느 단어를 문서에서 인식하여 추출하는 기법.  
추출된 개체명은 인명, 지명, 기관명, 시간 등으로 분류된다.  

## BIO 태깅
여러개의 단어 또는 토큰을 하나로 묶기 위해 도입된 태깅 시스템.  
B-begin 은 개체명의 시작  
I-inside 는 개체명의 중간 또는 끝  
O-outside 는 개체명이 아닌것을 나타낸다.  
![image](https://user-images.githubusercontent.com/87703352/160055532-5ba017cc-77e5-4657-bed5-0a83ef4e2169.png)

# 데이터 예시
## KLUE-NER
Label은 Person(PS), Location(LC), Organization(OG), Date(DT), Time(TI), Quantity(QT) 이렇게 6종류가 있고,  
Character level BIO 태깅으로 되어있다.  
|Text|수|학| |A|형|의| |1|등|급||커|트|라|인|은|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Label|O|O|O|O|O|O|O|B-QT|I-QT|I-QT|O|O|O|O|O|O|
## CoNLL2003
Person (PER), Organization(ORG), Location(LOC), Miscellaneous (MISC)이렇게 4종류가 있고,
Word level BIO 태깅으로 되어있다.
|EU|rejects|German|call|to|boycott|British|lamb|.|
|---|---|---|---|---|---|---|---|---|
|B-ORG|O|B-MISC|O|O|O|B-MISC|O|O|
