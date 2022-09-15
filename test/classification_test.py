

import toml
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append("../")
from model.textcnn import TextCNNConfig, TextCNN
from utils.data import NLPDataset, ClassificationBasePreProcessor, NLPCollator
from common.module.trainer import Trainer
from common.module.evaluate import Evaluator
from transformers import BertTokenizerFast


with open("../common/configs/task.conf.toml", "r", encoding="utf-8") as f:
    task_config = toml.load(f)
print(task_config)

tokenizer = BertTokenizerFast("../recources/bert_vocab.txt")

# 读取数据
train_data = NLPDataset("iPhone/train.json", ClassificationBasePreProcessor(tokenizer=tokenizer, **task_config), tokenizer=tokenizer)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=NLPCollator.classificationBaseCollateFn)
valid_data = NLPDataset("iPhone/dev.json", ClassificationBasePreProcessor(tokenizer=tokenizer, **task_config), tokenizer=tokenizer, is_predict=True)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=NLPCollator.classificationBaseCollateFn)
test_data = NLPDataset("iPhone/test.json", ClassificationBasePreProcessor(tokenizer=tokenizer, **task_config), tokenizer=tokenizer, is_predict=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=NLPCollator.classificationBaseCollateFn)

model = TextCNN(TextCNNConfig.from_config_file("../common/configs/textcnn.conf.toml"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(train_loader, valid_loader, model, optimizer, nn.CrossEntropyLoss(), test_loader=test_loader, ckpt_path="ckpt/textcnn.ckpt", device="cpu", early_stop=True)

trainer.train(10)


evaluator = Evaluator(model, 'classification', task_config['label_list'], device="cpu")
evaluator._classification_evaluate(test_loader)