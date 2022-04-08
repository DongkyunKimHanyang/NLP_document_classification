# MRC 란?
Machine Reading Comprehension 기계독해는 기계가 주어진 지문(Context)을 읽고 주어진 질문 (Question)에 답 (Answering)하는 task 세팅이다. 주로 QA (Question & Answering) 문제를 해결하는데 사용되며,
딥러닝 기반의 자연어처리 기술이 등장하면서 성능이 크게 개선되었다.  

# 대표적인 Dataset
## The Stanford Question and Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/))
SQuAD는 위키피디아 지문을 기반으로 생성된 QA 데이터셋으로, Crowdworker들이 위키피디아 지문과 그에대한 질문과 답변을 표기해놓은 벤치마크 데이터셋이다.  
SQuAD 2.0에서는 지문에서 답변을 찾을 수 없는 Unanswerable 데이터셋이 추가되었다.  
  
SQuAD 2.0 데이터 예시 JSON 형태로 되어있다.  
``` python
{
  "version": "v2.0", // 버전
  "data": [
    {
      "title": "Normans", // Article 제목
      "paragraphs": [
        {
          "qas": [
            {
              "question": "In what country is Normandy located?", // 질문
              "id": "56ddde6b9a695914005b9628",  
              "answers": [
                {
                  "text": "France", // Answer
                  "answer_start": 159 // Answer 문자의 시작위치
                },
                {
                  "text": "France",
                  "answer_start": 159
                }
              ],
              "is_impossible": false  // Unanswerable 표시
            }
          ]
        },
        {
          "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy ...."
        }
      ]
    }
  ]
}
```

## Korean QA Dataset ([KorQuAD](https://korquad.github.io/))
LG CNS에서 공개한 QA 데이터셋으로, 한국 위키피디아 지문을 기반으로 사람들이 질문과 답변을 달아놓았다. 총 7만개 이상의 샘플을 포함하고 있다.  
KorQuAD 2.0은 KorQuAD 1.0에서 2만여개의 샘플을 추가했고 답변을 찾기 위해 1문단이 아닌 문서 전체에서 찾아야 하도록 구성되어있다. 또한 표와 리스트도 포함되어있다.  
KorQuAD 1.0의 데이터의 구조는 SQuAD와 비슷하다.  
  
## KLUE-MRC
Korean Language Understanding Evaluation([KLUE](https://klue-benchmark.com/))는 8가지의 task를 포함하고 있는 벤치마크 데이터셋인데 MRC 또한 포함되어있다.  
데이터의 구조는 SQuAD 2.0과 유사하다.  
