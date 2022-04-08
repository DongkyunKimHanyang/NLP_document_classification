import torch
from torch.utils import data

import json
import math
import collections
import os
from tqdm import tqdm
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features
from transformers.data.processors.utils import InputFeatures

class MRCdataset():
    def __init__(self,input_file,tokenizer,max_seq_length,doc_stride,max_query_length,cache_file):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.id_key_name = "guid" if "klue" in input_file else "id"
        if "train" in cache_file:
            self.is_training = True
        else:
            self.is_training = False

        if os.path.exists(cache_file+'_examples') == False:
            reader = open(input_file,"r",encoding="utf-8")
            input_data = json.load(reader)["data"]
            self.examples = self._create_examples(input_data)
            torch.save(self.examples,cache_file+'_examples')
        else:
            self.examples = torch.load(cache_file+'_examples')

        if os.path.exists(cache_file+'_features') == False:
            self.features = self._create_features(self.examples)
            torch.save(self.features, cache_file+'_features')
        else:
            self.features = torch.load(cache_file+'_features')



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_masks = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(
            0 if feature.token_type_ids is None else feature.token_type_ids,
            dtype=torch.long,
        )

        cls_indexes = torch.tensor([feature.cls_index], dtype=torch.long)
        p_masks = torch.tensor([feature.p_mask], dtype=torch.float)
        is_impossibles = torch.tensor([feature.is_impossible], dtype=torch.float)
        if feature.is_impossible:
            a=1
        if self.is_training:
            start_positions = torch.tensor([feature.start_position], dtype=torch.long)
            end_positions = torch.tensor([feature.end_position], dtype=torch.long)
            return input_ids, attention_masks, token_type_ids, start_positions, end_positions, cls_indexes, p_masks, is_impossibles, idx
        else:
            return input_ids, attention_masks, token_type_ids, cls_indexes, p_masks, is_impossibles, idx


    def _create_examples(self,input_data):
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa[self.id_key_name]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if self.is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples

    def _create_features(self, examples):
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=self.is_training,
            return_dataset=False,
            threads=10,
        )
        return features
