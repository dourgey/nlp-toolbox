import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler

from model.textcnn import TextCNNConfig, TextCNN
from utils.data import NLPDataset, ClassificationBasePreProcessor, NLPCollator


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
                 optimizer: torch.optim.Optimizer, criterion,
                 test_loader, ckpt_path, early_stop=False, early_stop_patience=7, evaluate_fn=None, device='cpu', auto_mixed_precision=True, **kwargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.test_loader = test_loader
        self.ckpt_path = ckpt_path
        self.early_stop = early_stop
        self.device = device
        self.total_step = 0
        self.epoch_num = 0
        self.model.to(device)
        self.evaluate_fn = evaluate_fn
        self.auto_mixed_precision = auto_mixed_precision

        self.early_stop = early_stop
        if early_stop:
            self.es = EarlyStopping(early_stop_patience)

    def train(self, epoch_num):
        self.model.train()
        scaler = GradScaler()
        for epoch in tqdm.tqdm(range(epoch_num)):
            self.epoch_num += 1
            torch.cuda.empty_cache()   # 清理GPU缓存
            for x, y in tqdm.tqdm(self.train_loader):
                self.total_step += 1
                x = [_x.to(self.device) for _x in x]
                y = y.to(self.device)

                # forward
                if self.auto_mixed_precision and self.device == 'cuda':
                    with autocast():
                        y_pred = self.model(x)
                        loss = self.criterion(y_pred, y)

                    # Backward and optimize
                    # Scales loss. 为了梯度放大.
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)

                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if self.early_stop:
                with torch.no_grad():
                    if self.evaluate_fn is None:
                        val_loss = torch.mean(torch.tensor([self.criterion(self.model([_x.to(self.device) for _x in x]), y.to(self.device)) for x, y in self.valid_loader])).cpu()
                    else:
                        val_loss = self.evaluate_fn(self.model, self.valid_loader)
                    early_stop_status = self.es(val_loss)
                    if early_stop_status == EARLY_STOP_STATUS.NEED_SAVE_MODEL:
                        self._save_cpkt()
                    elif early_stop_status == EARLY_STOP_STATUS.NEED_EARLY_STOP:
                        print("======= EARLY STOPPING  ========")
                        break
                    else:
                        continue

    def _save_cpkt(self):
        """
        保存检查点 save checkpoint
        :return: None
        """
        torch.save({
            'epoch': self.epoch_num,
            'step': self.total_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.ckpt_path)

    def _load_ckpt(self):
        """
        加载检查点 load checkpoint
        :return: None
        """
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict']).to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch_num = checkpoint['epoch']
        self.total_step = checkpoint['step']


if __name__ == '__main__':
    from transformers import BertTokenizer

    train_file = '../test/iPhone/train.json'
    tokenizer = BertTokenizer.from_pretrained('../../test/bert_www')
    preprocessor = ClassificationBasePreProcessor(label_list=["0", "1", "2"], max_seq_length=128, tokenizer=tokenizer)

    train_set = NLPDataset('../../test/iPhone/train.json', preprocessor, tokenizer)
    valid_set = NLPDataset('../../test/iPhone/dev.json', preprocessor, tokenizer)
    train_loader = DataLoader(train_set, batch_size=64, collate_fn=NLPCollator.classificationBaseCollateFn, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=64, collate_fn=NLPCollator.classificationBaseCollateFn, shuffle=False)
    config = TextCNNConfig.from_config_file('../configs/textcnn.conf')
    print(config)
    model = TextCNN(config)
    trainer = Trainer(train_loader, valid_loader, model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), criterion=nn.CrossEntropyLoss(), test_loader=None, ckpt_path='../../test/ckpt/textcnn.ckpt', device='cuda', early_stop=True)
    trainer.train(epoch_num=100)