# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from Dataset import MRCdataset

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult

import argparse
import os
import json
from tqdm import tqdm




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default="korquad_1.0", choices=["squad_2.0", "klue-mrc-v1.1","korquad_1.0"])
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--dev_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_rate", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--w_decay", type=float, default=0.001)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    train_cache_file = f'./Data/{args.dataset}/cached_train'
    dev_cache_file = f'./Data/{args.dataset}/cached_dev'

    output_dir = f'./Result/{args.dataset}'
    os.makedirs(output_dir,exist_ok=True)


    if args.dataset == "squad_2.0":
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        MRC_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased").to(device)
        train_dataset = MRCdataset('./Data/squad_2.0/train-v2.0.json',tokenizer,args.max_seq_length, args.doc_stride,args.max_query_length, train_cache_file)
        dev_dataset = MRCdataset('./Data/squad_2.0/dev-v2.0.json', tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, dev_cache_file)
        version_2_with_negative = True
    elif args.dataset == "klue-mrc-v1.1":
        tokenizer = BertTokenizer.from_pretrained("klue/roberta-base")
        MRC_model = AutoModelForQuestionAnswering.from_pretrained("klue/roberta-base").to(device)
        train_dataset = MRCdataset('./Data/klue-mrc-v1.1/klue-mrc-v1.1_train.json', tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, train_cache_file)
        dev_dataset = MRCdataset('./Data/klue-mrc-v1.1/klue-mrc-v1.1_dev.json', tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, dev_cache_file)
        version_2_with_negative = True
    elif args.dataset == "korquad_1.0":
        tokenizer = BertTokenizer.from_pretrained("klue/roberta-base")
        MRC_model = AutoModelForQuestionAnswering.from_pretrained("klue/roberta-base").to(device)
        train_dataset = MRCdataset('./Data/korquad_1.0/KorQuAD_v1.0_train.json', tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, train_cache_file)
        dev_dataset = MRCdataset('./Data/korquad_1.0/KorQuAD_v1.0_dev.json', tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, dev_cache_file)
        version_2_with_negative = False

    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True)
    dev_dataloader = DataLoader(dev_dataset,batch_size=args.dev_batch_size,shuffle=False)

    optimizer = optim.AdamW(MRC_model.parameters(), args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(len(train_dataloader) * args.warmup_rate)*args.n_epochs,num_training_steps=len(train_dataloader)*args.n_epochs)
    dev_eval_str = ""

    for epoch in range(1, args.n_epochs + 1):
        train_bar = tqdm(train_dataloader)
        dev_results = []

        MRC_model.train()
        for i, features in enumerate(train_bar):
            input_ids, attention_masks, token_type_ids, start_positions, end_positions, cls_indexes, p_masks, is_impossibles, idx = features
            input_ids, attention_masks, token_type_ids, start_positions, end_positions = input_ids.to(device), attention_masks.to(device), token_type_ids.to(device), start_positions.to(device), end_positions.to(device)
            outputs = MRC_model(input_ids,attention_masks,start_positions=start_positions,end_positions=end_positions)
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            MRC_model.eval()
            dev_bar = tqdm(dev_dataloader)
            for i, features in enumerate(dev_bar):
                input_ids, attention_masks, token_type_ids, cls_indexes, p_masks, is_impossibles, idx = features
                input_ids, attention_masks, token_type_ids, cls_indexes, p_masks, is_impossibles, idx = input_ids.to(device), attention_masks.to(device), token_type_ids.to(device), cls_indexes.to(device), p_masks.to(device), is_impossibles.to(device), idx.to(device)
                outputs = MRC_model(input_ids, attention_masks)
                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]

                batch_results = list()
                for j, feature_index in enumerate(idx):
                    unique_id = train_dataset.features[feature_index].unique_id
                    single_example_start_logits = start_logits[j].tolist()
                    single_example_end_logits = end_logits[j].tolist()
                    dev_results.append(SquadResult(unique_id,single_example_start_logits,single_example_end_logits))

            compute_predictions_logits(
                all_examples=dev_dataset.examples,
                all_features=dev_dataset.features,
                all_results=dev_results,
                n_best_size=20,
                max_answer_length=30,
                do_lower_case=False,
                output_prediction_file=output_dir+'/dev_predictions.json',
                output_nbest_file=None,
                output_null_log_odds_file=output_dir+'/dev_null_log_odd.json',
                verbose_logging=False,
                version_2_with_negative=version_2_with_negative,
                null_score_diff_threshold=1.0,
                tokenizer=tokenizer,
            )
            dev_preds = json.load(open(output_dir+'/dev_predictions.json',"r",encoding="utf-8"))
            if version_2_with_negative:
                null_odds = json.load(open(output_dir+'/dev_null_log_odd.json',"r",encoding="utf-8"))
            else:
                null_odds = None
            dev_eval_dict = squad_evaluate(dev_dataset.examples,dev_preds,null_odds)
            dev_eval_str += str(dev_eval_dict) + "\n\n\n"
    dev_eval_file = open(output_dir+'/dev_eval.txt','w')
    print(dev_eval_str,file=dev_eval_file)
    dev_eval_file.close()
