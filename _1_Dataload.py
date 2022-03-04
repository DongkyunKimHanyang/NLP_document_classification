import torch
from torch.utils import data

import numpy as np

from datasets import load_dataset
from transformers import RobertaTokenizer
class IMDBdataset(data.Dataset):
    def __init__(self, is_train=True):
        self.IBDM = load_dataset('imdb', split='train') if is_train else load_dataset('imdb', split='test')#IMDB로부터 데이터를 불러옵니다.
        self.text = self.IBDM['text']
        self.label = self.IBDM['label']
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')# Hugging Face에 올라와있는 Roberta의 토크나이저입니다.

    def __len__(self):
        return len(self.IBDM)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]
        #Roberta는 최대 입력길이가 512입니다. 따라서 길이 512가 넘어가는 데이터는 truncate합니다.
        features = self.tokenizer(self.text[idx],max_length = 512, truncation=True)
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        return input_ids, attention_mask, label

    def pad(self,batch):
        input_ids = [sample[0] for sample in batch]
        attention_mask = [sample[1] for sample in batch]
        label = [sample[2] for sample in batch]
        max_len = 512
        #최대 입력길이인 512까지 padding을 넣어줍니다.
        pad_f = lambda seq, max_len, value: torch.tensor([sample + [value]*(max_len-len(sample)) for sample in seq],dtype=torch.int64)
        input_ids = pad_f(input_ids, max_len, self.tokenizer.pad_token_type_id) #input_ids는 토크나이저의 pad 토큰의 값을 padding 해줍니다.
        attention_mask = pad_f(attention_mask, max_len, 0)#attention_mask는 0을 padding 해줍니다.
        return input_ids, attention_mask, torch.tensor(label,dtype=torch.int64)