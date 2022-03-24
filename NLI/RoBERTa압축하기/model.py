import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from transformers import RobertaModel, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaLayer

from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import permutations
import math
import copy

from Dataload import Dataload
from custom_layers import CLS_head, Transformer_head



class NLI_middel_layer_model(nn.Module):
    def __init__(self,middle_index):
        super(NLI_middel_layer_model,self).__init__()
        self.Backbone_model = RobertaModel.from_pretrained('./klue/roberta-base',add_pooling_layer=False,output_hidden_states=True)
        self.Backbone_model.encoder.layer = self.Backbone_model.encoder.layer[:middle_index-1]
        self.config = self.Backbone_model.config

        self.classifier_head = CLS_head(self.config)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.Backbone_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.classifier_head(encoder_outputs)
        return logits

class NLI_online_distill_model(nn.Module):
    def __init__(self):
        super(NLI_online_distill_model,self).__init__()
        self.Backbone_model = RobertaModel.from_pretrained('./klue/roberta-base',add_pooling_layer=False,output_hidden_states=True)
        self.config = self.Backbone_model.config

        self.classifier_head = nn.ModuleList([CLS_head(self.config) for l in range(6)])


    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.Backbone_model(input_ids=input_ids, attention_mask=attention_mask).hidden_states
        logits = torch.cat([self.classifier_head[i](encoder_outputs[-6+i][:,0:1,:]) for i in range(6)],1)

        return logits





def load_NLI_model(name='baseline',**kwargs):
    if name =='baseline':
        return NLI_base_model()
    if name == "middel_layer":
        return NLI_middel_layer_model(**kwargs)
    if name == "distill":
        return NLI_online_distill_model(**kwargs)