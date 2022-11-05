import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv(r"data/train.csv")
    df.loc[:, "kfold"] = -1
    df.sample(frac=1).reset_index(drop=True)
    targets = df["label"].values

    skf = StratifiedKFold(n_splits=5)
    for fold, (train, val) in enumerate(skf.split(X=df, y=targets)):
        df.loc[val, "kfold"] = fold
    df.to_csv(r"data/folds.csv", index=False)
