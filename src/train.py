import os
import utils
import config
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm,trange
from transformers import AutoModel

def run_training(fold, params):
    df = pd.read_csv(r"data/folds.csv")
    feature_columns = ['premise', 'hypothesis']
    target_columns = ['label']

    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    val_df = df[df['kfold'] == fold].reset_index(drop=True)

    xtrain = train_df[feature_columns]
    ytrain = train_df[target_columns]
    xvalid = val_df[feature_columns]
    yvalid = val_df[target_columns]

    checkpoint = params['checkpoint']
    if checkpoint == 'xlm-roberta-base':
        input_size = 768
    elif checkpoint == 'bert-base-multilingual-cased':
        input_size = 768
    elif checkpoint == 'sentence-transformers/paraphrase-MiniLM-L6-v2':
        input_size = 384

    tokenizer = utils.AutoTokenizer.from_pretrained(checkpoint)
    train_dataloader = utils.MDWDataset(
        features=xtrain, targets=ytrain, tokenizer=tokenizer, device=config.device)
    val_dataloader = utils.MDWDataset(
        features=xvalid, targets=yvalid, tokenizer=tokenizer, device=config.device)

    train_loader = torch.utils.data.DataLoader(
        train_dataloader, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataloader, batch_size=params['val_batch_size'], shuffle=True)

    base_model = AutoModel.from_pretrained(checkpoint)
    model = utils.Model(model=base_model,
                  dropout=params['dropout'], linear_input_size=input_size)
    model.to(config.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=params['lr'])
    eng = utils.Engine(model=model, device=config.device, optimizer=optimizer)

    best_f1 = -np.inf
    best_all_metric = None
    early_stopping_iter = 2
    early_stopping_counter = 0

    for epoch in trange(config.epochs):
        all_metric_tr, f1_tr = eng.train(train_loader)
        all_metric_vl, f1_vl = eng.evaluate(val_loader)

        print(f'\nepoch_{epoch}_results:')
        print(f"""
                fold: {fold}
                epoch: {epoch}
                f1: {all_metric_vl['f1']}
                accuracy: {all_metric_vl['accuracy']}
                precision: {all_metric_vl['precision']}
                recall: {all_metric_vl['recall']}
                avg_loss_per_batch: {all_metric_vl['avg_loss_per_batch']}
                confusion_matrix: 
                \n{all_metric_vl['confusion_matrix']}
        """)

        if f1_vl > best_f1:
            best_f1 = f1_vl
            best_all_metric = all_metric_vl
            best_model = model.state_dict()
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break

    print(f'\nfold_{fold}_results:')
    print(best_all_metric)

    return best_f1,best_model
