import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel

import pandas as pd
import math



class CLS_head(nn.Module):
    def __init__(self,config):
        super(CLS_head,self).__init__()
        self.config = config

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.output_proj = nn.Linear(self.config.hidden_size,3)

    def forward(self,encoder_outputs):
        outputs = self.dropout(encoder_outputs)
        outputs = self.dense(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        logits = self.output_proj(outputs)
        return logits


class Transformer_head(nn.Module):
    def __init__(self,config):
        super(Transformer_head,self).__init__()
        #Roberta-base의 transformer layer를 따라서 구현한 클래스 입니다..
        self.num_heads = 12
        self.attention_head_size = 64

        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.Wo = nn.Linear(config.hidden_size,config.hidden_size)
        self.LayerNorm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ff1 = nn.Linear(config.hidden_size,config.intermediate_size)
        self.ff2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dense_classifier = CLS_head(config)

    def forward(self,hidden_state,attention_mask):
        query = self.query(hidden_state)
        key = self.key(hidden_state)
        value = self.value(hidden_state)

        query = self.transepose_hidden(query)
        key = self.transepose_hidden(key)
        value = self.transepose_hidden(value)

        #Scaled dot product attention을 수행해 줍시다.
        attention_score = torch.matmul(query,key.transpose(-1,-2)) / math.sqrt(self.attention_head_size) #  (batch, num_heads, Seq_len, Seq_len) 멀티 헤드로 잘라 줍니다.
        attention_mask = (1 - attention_mask[:,None,None,:]) * -10000 #  (batch, 1, 1, Seq_len)
        attention_score = attention_score + attention_mask
        attention_score = torch.softmax(attention_score,-1)

        context_vector = torch.matmul(attention_score, value) # 어텐션 레이어의 context 벡터가 완성되었습니다. (batch, num_heads, Seq_len, attention_hidden_size)
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous() #(batch, Seq_len, num_heads, attention_hidden_size)
        context_vector = context_vector.view(*hidden_state.size()) #(batch, Seq_len, hidden_size)로 원상 복귀

        attention_outputs = self.Wo(context_vector) #Wo 레이어를 거치고
        attention_outputs = self.dropout(attention_outputs)
        attention_outputs = self.LayerNorm_1(attention_outputs + hidden_state) #residual connection까지 해줍니다.

        #feed-forward 연산
        outputs = self.ff1(attention_outputs)
        outputs = torch.nn.functional.gelu(outputs)
        outputs = self.ff2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm_2(outputs + attention_outputs)

        logits = self.dense_classifier(outputs[:,0,:])#첫번째 토큰만 사용해줍시다.
        return outputs, logits

    def transepose_hidden(self, hidden_state):
        #hidden_vector의 차원은 (Batch, Seq_len, hidden_size=768) 입니다.
        #멀티 헤드 어텐션 구현을 위해서
        #이것을 (Batch,num_heads=12,Seq_len,attention_hidden_size=64)로 변환 해줍니다.
        new_shape = hidden_state.size()[:-1] + (self.num_heads, self.attention_head_size)
        hidden_state = hidden_state.view(*new_shape)
        return hidden_state.permute(0,2,1,3)