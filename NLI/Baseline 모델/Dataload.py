import pandas as pd
import numpy as np
import math
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split

class Dataload(data.Dataset):
    def __init__(self,dataset, tokenizer):

        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self,idx):
        #premise와 hypothesis를 토크나이징하고 [SEP] 으로 이어 붙입니다.
        example = self.dataset.iloc[idx]
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        labels = example["label"]

        features = self.tokenizer(premise, hypothesis)
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        #Label을 인코딩 해줍시다.
        if labels == 'entailment':
            labels = 0
        elif labels == 'contradiction':
            labels = 1
        elif labels == 'neutral':
            labels = 2
        return input_ids, attention_mask, labels


    def pad(self,batch): #총 128의 길이 까지 Padding해줍니다.
        f = lambda x: [sample[x] for sample in batch]
        seq_len = [len(sample[0]) for sample in batch]
        input_ids = f(0)
        attention_mask = f(1)
        labels = f(2)
        max_len = 128

        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, max_len)
        attention_mask = padding(attention_mask, 0, max_len)
        labels = torch.tensor(labels,dtype=torch.int64)
        return input_ids, attention_mask, labels


