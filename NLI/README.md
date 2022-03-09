# NLI란
자연어추론_Natural Language Inference는 제시된 전제(premise)에 대한 가설(Hypothesis)이 참(Entailment)인지 거짓(Contradiction)인지 판단 불가능(Neutral)인지 분류하는 문제입니다.  

아래는 [KLUE의 NLI 데이터](https://klue-benchmark.com/tasks/68/overview/description) 예시입니다.  
|premise|hypothesis|label|
|---|---|---|
|씨름은 상고시대로부터 전해져 내려오는 남자들의 대표적인 놀이로서, 소년이나 장정들이 넓고 평평한 백사장이나 마당에서 모여 서로 힘과 슬기를 겨루는 것이다.|씨름의 여자들의 놀이이다.|Contradiction|
|이번 증설로 코오롱인더스트리는 기존 생산량 7만7000톤에서 1만6800톤이 늘어나 총 9만 3800톤의 생산 능력을 확보하게 됐다.|코오롱 인더스트리는 총 9만 3800톤의 생산 능력을 확보했다.|Entailment|
|자신뿐만 아니라 남을 돕고자 하는 청년의 꿈과 열정에 모두가 주목하고 있다.|모든 청년은 꿈과 열정을 가지고 있다.|Neutral|
