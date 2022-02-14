import sys
sys.path.append("../")

from utils.data import NLPDataset, ClassificationBasePreProcessor
from model.bert_cnn import *
from transformers import BertTokenizer, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from sklearn.metrics import classification_report, precision_score, \
    recall_score, f1_score, accuracy_score, precision_recall_fscore_support
device = "cuda"


# 评价标准
def evaluate(model, test_loader):
    model.eval()
    model.to(device)

    true_y = []  # 真实值
    pred_y = []  # 预测值
    label_score = []
    with torch.no_grad():
        for inputs, input_mask, segment_ids, targets in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            targets = targets
            x = {"input_ids": inputs, "attention_mask": input_mask}

            logits = model(x)
            label_score += torch.max(torch.softmax(logits, 1), dim=1)[0].cpu().tolist()
            y_pred = torch.max(torch.softmax(logits, 1), dim=1)[1]
            pred_y += y_pred.cpu().detach().tolist()
            true_y += targets.cpu().tolist()

        report = classification_report(true_y, pred_y)
        print(report)
        print("Test-Accuracy: ", accuracy_score(true_y, pred_y))
        print("Test-Precision: ", precision_score(true_y, pred_y, average='macro', labels=[0, 1, 2]))
        print("Test-Recall: ", recall_score(true_y, pred_y, average='macro', labels=[0, 1, 2]))
        print("Test-F1 Score: ", f1_score(true_y, pred_y, average='macro', labels=[0, 1, 2]))

    # result = ["\t".join([str(label), str(score)]) for label, score in zip(pred_y, label_score)]
    # with open('predict_result.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(result))

bert_vocab = r'D:\Works\codes\voice_quality\voice_quality_labeling\voice-quality-labeling-srv\target\classes\config_online\label_config\hbgprivatecall\house\bert_vocab.txt'
tokenizer = DistilBertTokenizerFast("../test/distilbert/vocab.txt")
train_file = "dataset/out_train.json"
valid_file = "dataset/out_dev.json"
test_file = "dataset/out_test.json"

config = BertCNNConfig(3, None, 4, [3, 4, 5], from_pretrain="../test/distilbert/")
model = DistillBertCNN(config).to(device)
optimizer = Adam(model.parameters())
critiron = torch.nn.CrossEntropyLoss()

preprocessor = ClassificationBasePreProcessor({"0": 0, "1": 1, "2": 2}, 128, tokenizer)
train_set = NLPDataset(train_file, preprocessor, tokenizer)
test_set = NLPDataset(train_file, preprocessor, tokenizer)

def collate_fn(batch):
    inputs = torch.tensor([x.inputs for x in batch], dtype=torch.int32)
    input_mask = torch.tensor([x.input_mask for x in batch], dtype=torch.int32)
    segment_ids = torch.tensor([x.segment_ids for x in batch], dtype=torch.int32)
    targets = torch.tensor([x.target for x in batch])

    return inputs, input_mask, segment_ids, targets  # .unsqueeze(1)


train_loader = DataLoader(train_set, 32, True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, 32, True, collate_fn=collate_fn)
epoch = 1
loss_list = []
for epoch in tqdm.tqdm(range(epoch)):
    model.train()
    for inputs, input_mask, segment_ids, targets in train_loader:
        inputs = inputs.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        targets = targets.to(device)
        x = {"input_ids": inputs, "attention_mask": input_mask}
        y_hat = model(x)
        loss = critiron(y_hat, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    evaluate(model, test_loader)