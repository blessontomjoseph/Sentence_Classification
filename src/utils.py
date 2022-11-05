import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm, trange


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
    def loss_fn(outputs, targets):
        loss = nn.CrossEntropyLoss()
        return loss(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for batch in tqdm(data_loader):
            self.optimizer.zero_grad()
            inputs = batch['x']
            targets = batch['y']
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss/len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for batch in tqdm(data_loader):
            inputs = batch['x']
            targets = batch['y']
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            final_loss += loss.item()
        return final_loss/len(data_loader)


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
