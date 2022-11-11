import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config
import os


def aug_interchange(data):
    data.reset_index(drop=True, inplace=True)
    new_data = pd.DataFrame()
    new_data['premise'] = pd.concat([data['premise'], data['hypothesis']])
    new_data['hypothesis'] = pd.concat([data['hypothesis'], data['premise']])
    new_data['lang_abv'] = pd.concat([data['lang_abv'], data['lang_abv']])
    new_data['label'] = pd.concat([data['label'], data['label']])
    aug_data = new_data.reset_index(drop=True)
    return aug_data


def load_data(train_path, test_path, more_path, augment=False, interchange=False):
    less_data = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    less_data.drop(['id', 'language'], axis=1, inplace=True)
    test.drop(['language'], axis=1, inplace=True)

    if augment:
        extra_data = os.listdir(more_path)
        more_data = pd.DataFrame(
            columns=['premise', 'hypothesis', 'lang_abv', 'label'])
        for data_file in extra_data:
            sub = pd.read_csv(os.path.join(more_path, data_file))
            sub.drop(['Unnamed: 0'], axis=1, inplace=True)
            more_data = pd.concat([more_data, sub])

        more_data.dropna(inplace=True)
        data = pd.concat([less_data, more_data])
        data.reset_index(drop=True, inplace=True)
    else:
        data = less_data

    if interchange:
        data = aug_interchange(data)
    else:
        pass
    return data, test


if __name__ == "__main__":
    data, test = load_data(train_path=r"data/train.csv",
                           test_path=r"data/test.csv",
                           more_path=r"data/aug_data",
                           augment=config.aug_translation,
                           interchange=config.aug_interchange)

    df = data.copy()
    df.loc[:, "kfold"] = -1
    df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    targets = df["label"].to_list()
    skf = StratifiedKFold(n_splits=config.n_splits)
    for fold, (train, val) in enumerate(skf.split(X=df, y=targets)):
        df.loc[val, "kfold"] = fold

    df.to_csv(r"folds/folds.csv", index=False)
