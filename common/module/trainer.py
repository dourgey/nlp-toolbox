import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, model: nn.Module,
                 optimizer: torch.optim.Optimizer, criterion, num_epoch,
                 test_loader, early_stop, device='cpu', **kwargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.test_loader = test_loader
        self.num_epoch = num_epoch
        self.early_stop = early_stop
        self.device = device
        self.total_step = 0
        self.model.to(device)



    def train(self):
        self.model.train()
        for epoch in tqdm.tqdm(range(self.num_epoch)):
            for x, y in self.train_loader:
                self.total_step += 1
                x = x.to(self.device)
                y = y.to(self.device)

                # forward
                y_hat = self.model(x)
                loss = self.criterion(y, y_hat)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        pass


    def test(self, data_loader):
        self.model.eval()


    def _save_cpkt(self):
        pass

    def _load_ckpt(self):
        pass
