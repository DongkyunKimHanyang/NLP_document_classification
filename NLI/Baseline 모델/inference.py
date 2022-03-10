import torch
from torch.utils import data

from model import NLI_model
from Dataload import Dataload

from transformers import RobertaModel, AutoTokenizer

from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="klue", choices=["klue,snli"])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if args.dataset == "klue":
        pre_train_name = "klue/roberta-large"
    else:
        pre_train_name = "roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(pre_train_name)

    test_dataset = pd.read_csv(f'./Data/{args.dataset}/test_data.csv').drop(columns=["index"])
    test_data = Dataload(test_dataset, tokenizer)
    test_batch = data.DataLoader(dataset=test_data,
                                  batch_size=128,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=test_data.pad)

    acc_file = open(f'./Result/{args.dataset}/test_fold_accuracy.txt', 'w')
    print("", file=acc_file)
    acc_file.close()
    
    soft_voting = np.zeros((len(test_data),3))
    label_array = np.array(['entailment', 'contradiction', 'neutral'])
    with torch.no_grad():
        acc_file = open(f'./Result/{args.dataset}/test_fold_accuracy.txt', 'a')
        for i in range(5):
            NLImodel = NLI_model(pre_train_name).to(device)
            NLImodel.load_state_dict(torch.load(f'./Result/{args.dataset}/Fold{i}model.pt'))
            logits = []
            labels = []
            dev_progress_bar = tqdm(test_batch)
            for i, batch in enumerate(dev_progress_bar):
                input_ids, attention_mask, label = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                logit, y_hat = NLImodel(input_ids, attention_mask)
                logits.extend(torch.nn.functional.softmax(logit, dim=1).detach().cpu().numpy())
                labels.extend(label.cpu().numpy())
            fold_pred = np.argmax(np.array(logits),-1)
            fold_pred = np.take(label_array,fold_pred)
            fold_acc = (test_dataset["label"].to_numpy()==fold_pred).sum()/len(test_dataset)
            print(f"Fold{i} {fold_acc:.4f}",file=acc_file)

            soft_voting += np.array(logits) / 5

    soft_voting = np.argmax(soft_voting,-1)
    soft_voting = np.take(label_array, soft_voting)

    sv_acc = (test_dataset["label"].to_numpy() == soft_voting).sum()/len(test_dataset)
    print(f"test accuracy {sv_acc:.4f}",file=acc_file)
    acc_file.close()

