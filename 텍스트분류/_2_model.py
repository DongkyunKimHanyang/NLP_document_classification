import torch
from torch import nn
from transformers import RobertaModel

import math

class dense_head(nn.Module): #2개의 Dense layer와 중간에 tanh 활성함수가 들어간 head 입니다. Roberta의 첫번째 토큰([CLS]) 벡터만 사용합니다.
    def __init__(self,config):
        super(dense_head,self).__init__()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermedicate_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_classifier = nn.Linear(config.hidden_size,2)

    def forward(self,last_hidden_state):
        outputs = self.dropout(last_hidden_state[:,0])
        outputs = self.intermedicate_dense(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense_classifier(outputs)
        return outputs

class avg_dense_head(nn.Module): #512개의 토큰 벡터를 가중 평균한뒤 Dense layer로 분류하는 Head입니다.
    def __init__(self,config):
        super(avg_dense_head,self).__init__()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.weight_layer = nn.Conv1d(in_channels=512,out_channels=1,kernel_size=1)
        self.dense_classifier = nn.Linear(config.hidden_size,2)

    def forward(self,last_hidden_state):
        outputs = self.dropout(last_hidden_state)
        outputs = self.weight_layer(outputs).squeeze()
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense_classifier(outputs)
        return outputs

class lstm_head(nn.Module):#512개의 토큰벡터를 LSTM에 입력한뒤 마지막 출력을 Dense layer로 분류하는 Head입니다.
    def __init__(self,config):
        super(lstm_head,self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm_layer = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size,num_layers=1,batch_first=True)
        self.dense_classifier = nn.Linear(config.hidden_size,2)

    def forward(self,last_hidden_state):
        batch_size = last_hidden_state.size(0)
        h0 = torch.randn(1, batch_size, self.config.hidden_size, device=last_hidden_state.device)
        c0 = torch.randn(1, batch_size, self.config.hidden_size, device=last_hidden_state.device)

        outputs = self.dropout(last_hidden_state)
        outputs, _ = self.lstm_layer(outputs,(h0,c0))
        outputs = torch.tanh(outputs[:,-1,:])
        outputs = self.dropout(outputs)
        outputs = self.dense_classifier(outputs)
        return outputs

class trainsformer_head(nn.Module):
    def __init__(self,config):
        super(trainsformer_head,self).__init__()
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

        self.dense_classifier = nn.Linear(config.hidden_size,2)

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
        outputs = self.LayerNorm_2(outputs + attention_outputs)[:,0,:] #첫번째 토큰만 사용해줍시다.

        outputs = self.dense_classifier(outputs)

        return outputs

    def transepose_hidden(self, hidden_state):
        #hidden_vector의 차원은 (Batch, Seq_len, hidden_size=768) 입니다.
        #멀티 헤드 어텐션 구현을 위해서
        #이것을 (Batch,num_heads=12,Seq_len,attention_hidden_size=64)로 변환 해줍니다.
        new_shape = hidden_state.size()[:-1] + (self.num_heads, self.attention_head_size)
        hidden_state = hidden_state.view(*new_shape)
        return hidden_state.permute(0,2,1,3)
    
class text_classification_model(nn.Module):
    #RoBERTa와 head를 종합하는 모델을 생성합니다.
    def __init__(self,head_name="dense"):
        super(text_classification_model,self).__init__()
        self.head_name = head_name

        self.back_bone = RobertaModel.from_pretrained('roberta-base',add_pooling_layer=False) #add_pooling_layer가 True면 Roberta 모델은 첫번째 토큰에 해당하는 벡터만 출력합니다.
        self.config = self.back_bone.config

        if self.head_name == "cls":
            self.head_layer = dense_head(self.config)
        elif self.head_name == "weight_avg":
            self.head_layer = avg_dense_head(self.config)
        elif self.head_name == "lstm":
            self.head_layer = lstm_head(self.config)
        elif self.head_name == "tr":
            self.head_layer = trainsformer_head(self.config)
    def forward(self,input_ids,attention_mask):
        outputs = self.back_bone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        if self.head_name == "tr":
            outputs = self.head_layer(outputs,attention_mask)
        else:
            outputs = self.head_layer(outputs)
        return outputs

