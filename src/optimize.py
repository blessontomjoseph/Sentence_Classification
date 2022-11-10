import pickle
import optuna
import numpy as np
import train
import config


def objective(trial):
    params = {
        'lr': trial.suggest_loguniform('lr', 1e-6, 1e-3),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.2),
        'train_batch_size': trial.suggest_categorical('train_batch_size', [16, 24]),
        'val_batch_size': trial.suggest_categorical('val_batch_size', [16, 24]),
        'checkpoint': trial.suggest_categorical('checkpoint', ['bert-base-multilingual-cased', 'xlm-roberta-base', 'sentence-transformers/paraphrase-MiniLM-L6-v2']),
    }

    f1 = []
    for fold in range(config.n_splits):
        f1_,_ = train.run_training(fold, params)
        f1.append(f1_)
    return np.mean(f1)


def optimize(objective):
    def things(objective):
        study = optuna.create_study(directions=['maximize'])
        study.optimize(objective, n_trials=config.optuna_n_trials)
        best_trial = study.best_trial
        return best_trial.values, best_trial.params
    
    for ep in range(5):
        score, params = things(objective=objective)
        if score[0] <= 0.5:
            continue
        else:
            break
    return score, params


if __name__=="__main__":
    score, params = optimize(objective)
    meta_data = {}
    meta_data['params'] = params
    meta_data['score'] = score[0]
    meta_data
    pickle.dump(meta_data, open('./meta_data.p','wb'))
    
