import torch
from torch import nn
from torch.utils import data
from transformers import RobertaModel, AutoTokenizer

from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import permutations
import math

from Dataload import Dataload


class CLS_head(nn.Module):
    def __init__(self,config):
        super(CLS_head,self).__init__()
        self.config = config

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.output_proj = nn.Linear(self.config.hidden_size,3)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,encoder_outputs):
        outputs = encoder_outputs.last_hidden_state[:,0,:]
        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        logits = self.output_proj(outputs)
        y_hat = logits.argmax(-1)
        return logits, y_hat

    def calc_loss(self,logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss


class NLI_model(nn.Module):
    def __init__(self,pre_train_name):
        super(NLI_model,self).__init__()

        self.Backbone_model = RobertaModel.from_pretrained(pre_train_name,add_pooling_layer=False,output_hidden_states=True)
        self.config = self.Backbone_model.config

        self.classifier_head = CLS_head(self.config)

    def forward(self,input_ids, attention_mask):
        #Roberta 출력일 Classifier head에 입력해 분류를 수행합니다.
        encoder_outputs = self.Backbone_model(input_ids=input_ids,attention_mask=attention_mask)
        logits, y_hat = self.classifier_head(encoder_outputs)
        return logits,  y_hat
