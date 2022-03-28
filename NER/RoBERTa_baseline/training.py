from Dataload import Conll_dataset, klue_dataset
from model import NER_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchcrf import CRF

from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report

import argparse
from tqdm import tqdm
import os
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str, default="klue", choices=["conll","klue"])
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_rate", type=float, default=1/5)
    parser.add_argument("--w_decay", type=float, default=0.001)
    parser.add_argument("--use_CRF", action='store_false')

    args = parser.parse_args()
    if args.use_CRF:
        result_dir = f'./Result/{args.dataset}/CRF'
    else:
        result_dir = f'./Result/{args.dataset}/Dense'
    os.makedirs(result_dir, exist_ok=True)
    record_file = open(result_dir + '/test_accuracy.txt', 'w')
    record_file.close()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    if args.dataset == 'conll':
        train_data, test_data = Conll_dataset("train"),  Conll_dataset("test")
        train_batch = data.DataLoader(train_data,
                                     batch_size=32,
                                     shuffle=True,
                                     collate_fn=train_data.pad)

        test_batch = data.DataLoader(test_data,
                                     batch_size=256,
                                     shuffle=False,
                                     collate_fn=test_data.pad)
        ignore_label_id = 9
        replace_ignore_label_to = 7
        pretrained_model_name = 'roberta-base'

    elif args.dataset == 'klue':
        train_data, test_data = klue_dataset("train"), klue_dataset("dev")
        train_batch = data.DataLoader(train_data,
                                      batch_size=32,
                                      shuffle=True,
                                      collate_fn=train_data.pad)
        test_batch = data.DataLoader(test_data,
                                     batch_size=256,
                                     shuffle=False,
                                     collate_fn=test_data.pad)

        ignore_label_id = 13
        replace_ignore_label_to = 0
        pretrained_model_name = 'klue/roberta-base'



    num_labels = ignore_label_id + 1
    model = NER_model(pretrained_model_name,num_labels).to(device)
    if args.use_CRF:
        crf = CRF(num_tags=num_labels, batch_first=True).to(device)
    else:
        ce_loss_fn = nn.CrossEntropyLoss()
    # AdamW를 적용합니다. 학습 파라미터중 bias와 Norm Layer의 weight에는 AdamW를 적용하지 않는것이 좋습니다.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.w_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    # Warmup 스케쥴러
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_batch) * args.warmup_rate) * args.n_epochs, num_training_steps=len(train_batch) * args.n_epochs)



    for epoch in range(1,args.n_epochs+1):
        model.train()
        crf.train()
        train_progress = tqdm(train_batch)
        for i, batch in enumerate(train_progress):
            input_ids, attention_mask, ner_labels = batch
            input_ids, attention_mask, ner_labels = input_ids.to(device), attention_mask.to(device), ner_labels.to(device)
            logits = model(input_ids,attention_mask)

            if args.use_CRF:
                loss= -crf(logits, ner_labels)
            else:
                loss = ce_loss_fn(logits.transpose(1,2),ner_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            crf.eval()
            record_file = open(result_dir + '/test_accuracy.txt', 'a')
            print(f"Epoch {epoch}", file=record_file)
            predictions = []
            labels = []
            dev_progress_bar = tqdm(test_batch)
            for i, batch in enumerate(dev_progress_bar):
                input_ids, attention_mask, ner_labels = batch
                input_ids, attention_mask, ner_labels = input_ids.to(device), attention_mask.to(device), ner_labels.to(device)
                logits = model(input_ids, attention_mask)

                if args.use_CRF:
                    logits = crf.decode(logits)
                    logits = np.array(logits)
                else:
                    logits = logits.cpu().numpy().argmax(-1)
                ner_labels = ner_labels.cpu().numpy()

                for logit, label in zip(logits,ner_labels):
                    pred = logit[label!=ignore_label_id]
                    pred = np.where(pred==ignore_label_id,replace_ignore_label_to,pred)
                    label = label[label!=ignore_label_id]

                    pred = np.take(test_data.label_list,pred).tolist()
                    label = np.take(test_data.label_list,label).tolist()
                    predictions.append(pred)
                    labels.append(label)

            print(classification_report(labels,predictions), file=record_file)
            record_file.close()
    torch.save(model.state_dict(),result_dir + '/model.pt')



