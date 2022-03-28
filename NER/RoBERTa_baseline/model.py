import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel


class CLS_head(nn.Module):
    def __init__(self,config,num_labels):
        super(CLS_head,self).__init__()
        self.config = config

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.output_proj = nn.Linear(self.config.hidden_size,num_labels)

    def forward(self,encoder_outputs):
        outputs = self.dropout(encoder_outputs)
        outputs = self.dense(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        logits = self.output_proj(outputs)
        return logits


class NER_model(nn.Module):
    def __init__(self,pretrained_name,num_labels,use_CRF=True):
        super(NER_model,self).__init__()
        self.Backbone_model = RobertaModel.from_pretrained(pretrained_name,add_pooling_layer=False)
        self.config = self.Backbone_model.config

        self.classifier_head = CLS_head(self.config,num_labels)

    def forward(self,input_ids,attention_mask):
        encoder_outputs = self.Backbone_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits= self.classifier_head(encoder_outputs)
        return logits


