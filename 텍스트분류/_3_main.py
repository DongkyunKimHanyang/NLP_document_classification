from _1_Dataload import IMDBdataset
from _2_model import text_classification_model

import torch
from torch.utils import data

import argparse
from tqdm import tqdm







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_name", type=str, default='weight_avg', choices=['cls', 'weight_avg', 'lstm'])
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_rate", type=float, default=0.3)
    parser.add_argument("--total_epochs", type=int, default=3)
    args = parser.parse_args()

    train_data = IMDBdataset(is_train=True)
    test_data = IMDBdataset(is_train=False)
    train_loader = data.DataLoader(dataset=train_data,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=train_data.pad)
    test_loader = data.DataLoader(dataset=test_data,
                                   batch_size=args.test_batch_size,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=test_data.pad)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = text_classification_model(args.head_name)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr = args.lr, steps_per_epoch = len(train_loader), epochs=args.total_epochs,pct_start=args.warmup_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1,args.total_epochs + 1):
        model.train()
        train_progress_bar = tqdm(train_loader)
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(train_progress_bar):
            input_ids, attention_mask, label = batch
            input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
            logits = model(input_ids,attention_mask)
            loss = loss_fn(logits,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred = logits.argmax(-1)
            acc = (pred == label).sum().item() / len(input_ids)
            train_loss = train_loss + (loss.item() - train_loss)/ (i + 1)
            train_acc = train_acc + (acc - train_acc) / (i + 1)

            train_progress_bar.set_description(f"Epoch {epoch}: Train loss {train_loss:.6f} train_acc {(train_acc * 100):.2f}%")

        with torch.no_grad():
            print("Calculating model accuracy on the testset")
            model.eval()
            num_correct = 0
            test_progress_bar = tqdm(test_loader)
            for i, batch in enumerate(test_progress_bar):
                input_ids, attention_mask, label = batch
                input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
                logits = model(input_ids, attention_mask)
                pred = logits.argmax(-1)
                num_correct += (pred == label).sum().item()
            print(f"Epoch {epoch}: Test accuracy = {(num_correct/len(test_data)*100):.2f}%")
