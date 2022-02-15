import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader


class EARLY_STOP_STATUS:
    NOTHING_TODO = 1
    NEED_SAVE_MODEL = 2
    NEED_EARLY_STOP = 3


class EarlyStopping:
    def __init__(self, patience):
        """
        Early stopping trick
        :param patience: How long to wait after last time validation loss improved. (epoch nums)
        """
        self.patience = patience
        self.count = 0
        self.val_loss_min = float("inf")

    def __call__(self, val_loss):
        if val_loss >= self.val_loss_min:
            self.count += 1
            if self.count == self.patience:
                return EARLY_STOP_STATUS.NEED_EARLY_STOP
        else:
            print(f"Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f})")
            self.val_loss_min = val_loss
            self.count = 0
            return EARLY_STOP_STATUS.NEED_SAVE_MODEL
        return EARLY_STOP_STATUS.NOTHING_TODO




class Trainer:
    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, model: nn.Module,
                 optimizer: torch.optim.Optimizer, criterion, num_epoch,
                 test_loader, ckpt_path, early_stop=False, early_stop_patience=7, evaluate_fn=None, device='cpu', **kwargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.test_loader = test_loader
        self.ckpt_path = ckpt_path
        self.num_epoch = num_epoch
        self.early_stop = early_stop
        self.device = device
        self.total_step = 0
        self.model.to(device)
        self.evaluate_fn = evaluate_fn

        self.early_stop = early_stop
        if early_stop:
            self.es = EarlyStopping(early_stop_patience)

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

            if self.early_stop:
                with torch.no_grad():
                    val_loss = self.evaluate_fn(self.model, self.valid_loader)
                    early_stop_status = self.es(val_loss)
                    if early_stop_status == EARLY_STOP_STATUS.NEED_SAVE_MODEL:
                        self._save_cpkt(epoch, loss)
                    elif early_stop_status == EARLY_STOP_STATUS.NEED_EARLY_STOP:
                        print("======= EARLY STOPPING  ========")
                        break
                    else:
                        continue

    def _save_cpkt(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.ckpt_path)

    def _load_ckpt(self):
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict']).to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
