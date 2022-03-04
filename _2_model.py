import torch
from torch import nn
from transformers import RobertaModel


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

class text_classification_model(nn.Module):
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

    def forward(self,input_ids,attention_mask):
        outputs = self.back_bone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        outputs = self.head_layer(outputs)
        return outputs

