import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from transformers import RobertaModel, AutoTokenizer, get_linear_schedule_with_warmup

from Dataload import Dataload
from model import NLI_model
from sklearn.model_selection import KFold

import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm



def training(args,pre_train_name,pad_fn,train_idx, valid_idx):
    #train fold와 valid fold를 만들어 줍니다.
    train_fold = torch.utils.data.Subset(train_data, train_idx)
    valid_fold = torch.utils.data.Subset(train_data, valid_idx)
    train_batch = data.DataLoader(dataset=train_fold,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=pad_fn)
    valid_batch = data.DataLoader(dataset=valid_fold,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=pad_fn)
    #모델을 불러와줍니다.
    NLImodel = NLI_model(pre_train_name).to(device)
    #AdamW를 적용합니다. 학습 파라미터중 bias와 Norm Layer의 weight에는 AdamW를 적용하지 않는것이 좋습니다.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in NLImodel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.w_decay},
        {'params': [p for n, p in NLImodel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    #Warmup 스케쥴러
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(len(train_batch) * args.warmup_rate)*args.n_epochs,num_training_steps=len(train_batch)*args.n_epochs)

    #training loop
    for epoch in range(1, args.n_epochs + 1):
        NLImodel.train()
        train_progress_bar = tqdm(train_batch)
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(train_progress_bar):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, y_hat = NLImodel(input_ids, attention_mask)
            loss = NLImodel.classifier_head.calc_loss(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc = (y_hat == labels).sum().data.cpu().numpy() / len(input_ids)
            train_loss = train_loss + (loss.item() - train_loss) / (i + 1)
            train_acc = train_acc + (acc - train_acc) / (i + 1)
            train_progress_bar.set_description(f"Epoch {epoch}: Train loss {train_loss:.6f} train_acc {(train_acc * 100):.2f}%")
    #validation에 대한 정확도 계산
    with torch.no_grad():
        NLImodel.eval()
        num_correct = 0.
        num_sample = 0.
        valid_acc = 0.
        valid_progress_bar = tqdm(valid_batch)
        for i, batch in enumerate(valid_progress_bar):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits, y_hat = NLImodel(input_ids, attention_mask)
            num_correct += (y_hat == labels).sum().data.cpu().numpy()
        valid_acc = (num_correct / len(valid_fold) * 100)

        acc_file = open(f'./Result/{args.dataset}/valid_fold_accuracy.txt', 'a')
        print(f"Fold {fold_idx}: {valid_acc:.4f}", file=acc_file)
        acc_file.close()
        torch.save(NLImodel.state_dict(), f"./Result/{args.dataset}/Fold{fold_idx}model.pt")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="klue", choices=["klue,snli"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_rate", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--w_decay", type=float, default=0.001)


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if args.dataset == "klue":
        pre_train_name = "klue/roberta-large"
    else:
        pre_train_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(pre_train_name)

    train_dataset = pd.read_csv(f'./Data/{args.dataset}/train_data.csv').drop(columns=["index"])
    train_data = Dataload(train_dataset, tokenizer)

    fold = KFold(n_splits=5, shuffle=True, random_state=0)

    acc_file = open(f'./Result/{args.dataset}/valid_fold_accuracy.txt', 'w')
    print("", file=acc_file)
    acc_file.close()
    #Train data을 5fold로 나누어 5번 각각 학습시킵니다.
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(train_data)))):
        training(args,pre_train_name,train_data.pad,train_idx, valid_idx)


