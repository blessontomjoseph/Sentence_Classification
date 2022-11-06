import utils
import config
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm,trange



def run_training(fold, params, save_model=False):
    df = pd.read_csv("./folds.csv")
    feature_columns = ['premise', 'hypothesis']
    target_columns = ['label']

    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    val_df = df[df['kfold'] == fold].reset_index(drop=True)

    xtrain = train_df[feature_columns]
    ytrain = train_df[target_columns]
    xvalid = val_df[feature_columns]
    yvalid = val_df[target_columns]

    train_dataloader = utils.MDWDataset(
        features=xtrain, targets=ytrain, tokenizer=config.tokenizer, device=config.device)
    val_dataloader = utils.MDWDataset(
        features=xvalid, targets=yvalid, tokenizer=config.tokenizer, device=config.device)

    train_loader = torch.utils.data.DataLoader(
        train_dataloader, batch_size=config.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataloader, batch_size=config.val_batch_size, shuffle=True)

    model = utils.Model(model=config.base_model, dropout=params['dropout'])
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
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
                fold:{fold},
                epoch:{epoch},
                {all_metric_vl},
        """)

        if f1_vl > best_f1:
            best_f1 = f1_vl
            best_all_metric = all_metric_vl
            if save_model:
                torch.save(model.state_dict(), f"model_{fold}.bin")
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break

    print(f'\nfold_{fold}_results:')
    print(best_all_metric)

    return best_f1
