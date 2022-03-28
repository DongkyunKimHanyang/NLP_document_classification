import numpy as np
import pandas as pd
import re
import torch
from torch.utils import data
import math
from datasets import load_dataset
from transformers import RobertaTokenizer, AutoTokenizer


class Conll_dataset(data.Dataset):
    def __init__(self,train_test="train"):
        self.dataset = pd.read_json(f"./Data/conll2003/{train_test}.json")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base",do_lower_case=False)
        self.label_list = np.array(['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])
        self.label_to_ids = {label: i for i, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset.iloc[idx]
        ner_labels = []
        tokens = []

        words = example["tokens"]
        ner_tags = example["ner_tags"]

        for word, ner_tag in zip(words, ner_tags):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            ner = [ner_tag] + [9] * (len(token) - 1)
            ner_labels.extend(ner)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        ner_labels = [self.label_to_ids["O"]] + ner_labels + [self.label_to_ids["O"]]

        assert len(input_ids) == len(attention_mask) == len(ner_labels)
        return input_ids, attention_mask, ner_labels

    def pad(self,batch):
        f = lambda x: [sample[x] for sample in batch]
        input_ids = f(0)
        attention_mask = f(1)
        ner_labels = f(2)
        max_len = max([len(sample) for sample in input_ids])

        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        input_ids = padding(input_ids,self.tokenizer.pad_token_id,max_len)
        attention_mask = padding(attention_mask,0,max_len)
        ner_labels = padding(ner_labels, 9, max_len)
        return input_ids, attention_mask, ner_labels



class klue_dataset(data.Dataset):
    def __init__(self,train_test="train"):
        raw_text = open(f'./Data/klue/klue-ner-v1.1_{train_test}.tsv','r').read().strip()
        self.dataset = re.split(r"\n\t?\n",raw_text)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base",do_lower_case=False)
        self.label_list = np.array(['O', 'B-PS', 'I-PS', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-DT', 'I-DT', 'B-TI', 'I-TI', 'B-QT', 'I-QT'])
        self.label_to_ids = {label: i for i, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx].split('\n')
        sentence = ""
        original_tags = []
        for line in example:
            if line.startswith('##'):
                continue
            character, character_ner_tag = line.split("\t")
            sentence += character
            if character != " ":
                original_tags.append(character_ner_tag)

        ner_labels = []
        tokens = []

        label_idx = 0
        for word in sentence.split(" "):
            if len(word) == 0:
                continue
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            ner = [self.label_to_ids[original_tags[label_idx]]] + [13] * (len(token) - 1)
            ner_labels.extend(ner)
            label_idx +=len(word)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        ner_labels = [self.label_to_ids["O"]] + ner_labels + [self.label_to_ids["O"]]

        assert len(input_ids) == len(attention_mask) == len(ner_labels)
        return input_ids, attention_mask, ner_labels

    def pad(self,batch):
        f = lambda x: [sample[x] for sample in batch]
        input_ids = f(0)
        attention_mask = f(1)
        ner_labels = f(2)
        max_len = max([len(sample) for sample in input_ids])

        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        input_ids = padding(input_ids,self.tokenizer.pad_token_id,max_len)
        attention_mask = padding(attention_mask,0,max_len)
        ner_labels = padding(ner_labels, 13, max_len)
        return input_ids, attention_mask, ner_labels





