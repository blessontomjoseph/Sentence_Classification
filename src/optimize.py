
import optuna
import numpy as np
import train

def objective(trial):
    params = {
        'lr': trial.suggest_loguniform('lr', 1e-6, 1e-3),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.7)
        
    }

    f1 = []
    for fold in range(5):
        f1_ = train.run_training(fold, params)
        f1.append(f1_)
    return np.mean(f1)


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)
    print('\nbest_trial')
    best_trial = study.best_trial
    print(best_trial.values)
    print(best_trial.params)

    f1 = []
    for fold in range(5):
        f1_ = train.run_training(fold, params=best_trial.params, save_model=True)
        f1.append(f1_)
    print(np.mean(f1))
