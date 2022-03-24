import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from transformers import RobertaModel, AutoTokenizer, get_linear_schedule_with_warmup

from Dataload import Dataload
from model import load_NLI_model

import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm



def train_middle_layer(args,train_batch,test_batch):
    # 모델을 불러와줍니다.
    acc_file = open(f'./Result/{args.model_name}/test_accuracy.txt', 'w')
    acc_file.close()
    for middle_index in range(1,13):
        print(f"Train transformer 1 ~ {middle_index} layers")
        NLImodel = load_NLI_model(args.model_name,middle_index=middle_index).to(device)
        loss_fn = nn.CrossEntropyLoss()
        # AdamW를 적용합니다. 학습 파라미터중 bias와 Norm Layer의 weight에는 AdamW를 적용하지 않는것이 좋습니다.
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in NLImodel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.w_decay},
            {'params': [p for n, p in NLImodel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        # Warmup 스케쥴러
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_batch) * args.warmup_rate) * args.n_epochs, num_training_steps=len(train_batch) * args.n_epochs)

        # training loop
        for epoch in range(1, args.n_epochs + 1):
            NLImodel.train()
            train_progress_bar = tqdm(train_batch)
            train_loss = 0
            train_acc = 0

            for i, batch in enumerate(train_progress_bar):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                logits = NLImodel(input_ids, attention_mask)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                y_hat = logits.argmax(-1)
                acc = (y_hat == labels).sum().data.cpu().numpy() / len(input_ids)
                train_loss = train_loss + (loss.item() - train_loss) / (i + 1)
                train_acc = train_acc + (acc - train_acc) / (i + 1)
                train_progress_bar.set_description(f"Epoch {epoch}: Train loss {train_loss:.6f} train_acc {(train_acc * 100):.2f}% lr {scheduler.get_last_lr()[0]}")
        with torch.no_grad():
            acc_file = open(f'./Result/{args.model_name}/test_accuracy.txt', 'a')
            logits = []
            labels = []
            dev_progress_bar = tqdm(test_batch)
            NLImodel.eval()
            for i, batch in enumerate(dev_progress_bar):
                input_ids, attention_mask, label = batch
                input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
                logit = NLImodel(input_ids, attention_mask)
                logits.extend(logit.detach().cpu().numpy())
                labels.extend(label.cpu().numpy())

            logits = np.array(logits)
            y_hat = np.argmax(logits, -1)
            labels = np.array(labels)
            acc = (labels == y_hat).sum() / len(labels)
            print(f"Middle layer {middle_index} accuracy =  {acc:.4f}", file=acc_file)
            acc_file.close()


def train_distill_model(args,train_batch,test_batch):
    # 모델을 불러와줍니다.
    acc_file = open(f'./Result/{args.model_name}/test_accuracy.txt', 'w')
    acc_file.close()
    train_acc_file = open(f'./Result/{args.model_name}/train_accuracy.txt', 'w')
    train_acc_file.close()

    NLImodel = load_NLI_model(args.model_name).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()
    # AdamW를 적용합니다. 학습 파라미터중 bias와 Norm Layer의 weight에는 AdamW를 적용하지 않는것이 좋습니다.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in NLImodel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.w_decay},
        {'params': [p for n, p in NLImodel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    # Warmup 스케쥴러
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_batch) * args.warmup_rate) * args.n_epochs, num_training_steps=len(train_batch) * args.n_epochs)

    # training loop
    for epoch in range(1, args.n_epochs + 1):
        NLImodel.train()
        train_progress_bar = tqdm(train_batch)
        train_loss = 0
        train_cls_loss = 0
        train_kd_loss = 0
        train_l2_loss = 0
        train_acc = 0

        for i, batch in enumerate(train_progress_bar):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = NLImodel(input_ids, attention_mask)
            cls_loss = ce_loss_fn(logits[:,-1], labels)

            kd_loss=0
            teacher_logits = torch.softmax(logits[:, -1].detach() / 3, -1)
            for j in range(5):
                cls_loss += ce_loss_fn(logits[:,j], labels)
                student_logits = torch.log_softmax(logits[:, j] / 3, -1)
                kd_loss += -(student_logits * teacher_logits).sum(1).mean()

            loss = 0.5 * cls_loss +  0.5 * kd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss = train_loss + (loss.item() - train_loss) / (i + 1)
            train_cls_loss = train_cls_loss + (cls_loss.item() - train_cls_loss) / (i + 1)
            train_kd_loss = train_kd_loss + (kd_loss.item() - train_kd_loss) / (i + 1)
            train_progress_bar.set_description(f"Epoch {epoch}: Total loss {train_loss:.6f}, CLS_loss {train_cls_loss:.6f}, KD_loss {train_kd_loss:.6f}")

        with torch.no_grad():
            train_acc_file = open(f'./Result/{args.model_name}/train_accuracy.txt', 'a')
            print(f"Epoch {epoch}",file=train_acc_file)
            logits = []
            labels = []
            train_progress_bar = tqdm(train_batch)
            NLImodel.eval()
            for i, batch in enumerate(train_progress_bar):
                input_ids, attention_mask, label = batch
                input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
                logit = NLImodel(input_ids, attention_mask)
                logits.extend(logit.detach().cpu().numpy())
                labels.extend(label.cpu().numpy())
            labels = np.array(labels)
            logits = np.array(logits)
            for j in range(6):
                y_hat = np.argmax(logits[:,j], -1)
                acc = (labels == y_hat).sum() / len(labels)
                print(f"Layer {j+7} accuracy =  {acc:.4f}", file=train_acc_file)
            train_acc_file.close()

        with torch.no_grad():
            acc_file = open(f'./Result/{args.model_name}/test_accuracy.txt', 'a')
            print(f"Epoch {epoch}",file=acc_file)
            logits = []
            labels = []
            dev_progress_bar = tqdm(test_batch)
            NLImodel.eval()
            for i, batch in enumerate(dev_progress_bar):
                input_ids, attention_mask, label = batch
                input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
                logit = NLImodel(input_ids, attention_mask)
                logits.extend(logit.detach().cpu().numpy())
                labels.extend(label.cpu().numpy())
            labels = np.array(labels)
            logits = np.array(logits)
            for j in range(6):
                y_hat = np.argmax(logits[:,j], -1)
                acc = (labels == y_hat).sum() / len(labels)
                print(f"Layer {j+7} accuracy =  {acc:.4f}", file=acc_file)
            acc_file.close()
    torch.save(NLImodel.state_dict(), f'./Result/{args.model_name}/model.pt')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type =str, default="distill", choices=["middel_layer","distill"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_rate", type=float, default=1/5)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--w_decay", type=float, default=0.001)


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)


    tokenizer = AutoTokenizer.from_pretrained("./klue/roberta-base")

    train_dataset = pd.read_csv(f'./Data/klue/train_data.csv').drop(columns=["index"])
    train_data = Dataload(train_dataset, tokenizer)
    train_batch = data.DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=train_data.pad)

    test_dataset = pd.read_csv(f'./Data/klue/test_data.csv').drop(columns=["index"])
    test_data = Dataload(test_dataset, tokenizer)
    test_batch = data.DataLoader(dataset=test_data,
                                 batch_size=128,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=test_data.pad)

    if args.model_name == "middel_layer":
        os.makedirs(f'./Result/{args.model_name}', exist_ok=True)
        train_middle_layer(args,train_batch,test_batch)
    if args.model_name == "distill":
        os.makedirs(f'./Result/{args.model_name}', exist_ok=True)
        train_distill_model(args,train_batch,test_batch)





