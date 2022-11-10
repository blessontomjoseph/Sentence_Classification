import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from tqdm.autonotebook import tqdm, trange
from sklearn import metrics


class MDWDataset:
    def __init__(self, features, targets, tokenizer, device):
        self.features = features
        self.targets = targets
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, items):
        feature_1 = list(self.features['premise'])[items]
        feature_2 = list(self.features['hypothesis'])[items]
        labels = torch.tensor(self.targets.iloc[items], dtype=torch.long)
        batch = self.tokenizer(text=feature_1, text_pair=feature_2, truncation=True,
                               padding='max_length', max_length=200, return_tensors='pt')

        return {
            'x': {k: v.squeeze(dim=0).to(self.device) for k, v in batch.items()},
            'y': labels.squeeze(dim=-1).to(self.device)
        }


class Engine:
    def __init__(self, model, device, optimizer):
        self.model = model
        self.sevice = device
        self.optimizer = optimizer

    @staticmethod
    def prediction(logits):
        _, preds = torch.max(f.softmax(logits.detach(), dim=-1), dim=-1)
        return preds

    @staticmethod
    def loss_fn(outputs, targets):
        loss = nn.CrossEntropyLoss()
        return loss(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        targets_full = np.array([])
        preds_full = np.array([])

        for batch in tqdm(data_loader):
            self.optimizer.zero_grad()
            inputs = batch['x']
            targets = batch['y']
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
            preds = self.prediction(outputs)
            preds_full = np.append(
                preds_full, preds.to('cpu').numpy(), axis=-1)
            targets_full = np.append(
                targets_full, targets.to('cpu').numpy(), axis=-1)

        stuff = {
            'f1': metrics.f1_score(targets_full, preds_full, average='weighted'),
            'accuracy': metrics.accuracy_score(targets_full, preds_full),
            'precision': metrics.precision_score(targets_full, preds_full, average='weighted'),
            'recall': metrics.recall_score(targets_full, preds_full, average='weighted'),
            'confusion_matrix': metrics.confusion_matrix(targets_full, preds_full),
            'avg_loss_per_batch': final_loss/len(data_loader)
        }

        return stuff, stuff['f1']

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        targets_full = np.array([])
        preds_full = np.array([])

        for batch in tqdm(data_loader):
            inputs = batch['x']
            targets = batch['y']
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            final_loss += loss.item()
            preds = self.prediction(outputs)
            preds_full = np.append(
                preds_full, preds.to('cpu').numpy(), axis=-1)
            targets_full = np.append(
                targets_full, targets.to('cpu').numpy(), axis=-1)

        stuff = {
            'f1': metrics.f1_score(targets_full, preds_full, average='weighted'),
            'accuracy': metrics.accuracy_score(targets_full, preds_full),
            'precision': metrics.precision_score(targets_full, preds_full, average='weighted'),
            'recall': metrics.recall_score(targets_full, preds_full, average='weighted'),
            'confusion_matrix': metrics.confusion_matrix(targets_full, preds_full),
            'avg_loss_per_batch': final_loss/len(data_loader)
        }

        return stuff, stuff['f1']


class Model(nn.Module):
    def __init__(self, model, dropout):
        super().__init__()
        self.model = model
        self.dropout = dropout
        self.out = nn.Sequential(nn.Dropout(p=self.dropout),
                                 nn.Linear(768, 3)
                                 )

    def forward(self, batch):
        output = self.model(**batch)
        output = output['pooler_output']
        return self.out(output)
