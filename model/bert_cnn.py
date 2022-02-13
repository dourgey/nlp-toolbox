import tqdm
from transformers import BertModel, BertConfig, BertTokenizerFast
import torch
import torch.nn as nn
from model_config import ModelConfig


class BertCNNConfig(ModelConfig):
    def __init__(self, num_classes, bert_config, num_filters, filter_sizes, dropout=0.1, from_pretrain=None):
        self.num_classes = num_classes
        self.bert_config = bert_config
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.from_pretrain = from_pretrain


class BertCNN(nn.Module):
    def __init__(self, config: BertCNNConfig):
        super(BertCNN, self).__init__()
        if config.from_pretrain is not None:
            bert_config = BertConfig.from_pretrained(config.from_pretrain)
            self.bert = BertModel(config=bert_config).from_pretrained(config.from_pretrain)
        else:
            self.bert = BertModel(config.bert_config)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 768)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x)).squeeze(3)
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.bert(**x).last_hidden_state
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class DistillBertCNN(nn.Module):
    def __init__(self, config: BertCNNConfig):
        super(DistillBertCNN, self).__init__()
        from transformers import DistilBertModel, DistilBertConfig
        if config.from_pretrain is not None:
            bert_config = DistilBertConfig.from_pretrained(config.from_pretrain)
            self.bert = DistilBertModel(config=bert_config).from_pretrained(config.from_pretrain)
        else:
            self.bert = DistilBertModel(config.bert_config)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 768)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x)).squeeze(3)
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.bert(**x).last_hidden_state
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


if __name__ == '__main__':
    from utils.data import NLPDataset, ClassificationBasePreProcessor
    from transformers import BertTokenizer, DistilBertTokenizerFast
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    device = "cuda"

    bert_vocab = r'D:\Works\codes\voice_quality\voice_quality_labeling\voice-quality-labeling-srv\target\classes\config_online\label_config\hbgprivatecall\house\bert_vocab.txt'
    tokenizer = DistilBertTokenizerFast("../test/distilbert/vocab.txt")
    train_file = "../test/dataset/ertai_train.json"
    valid_file = "../test/dataset/ertai_dev.json"
    test_file = "../test/dataset/ertai_test.json"

    config = BertCNNConfig(3, None, 4, [3, 4, 5], from_pretrain="../test/distilbert/")
    model = DistillBertCNN(config).to(device)
    optimizer = Adam(model.parameters())
    critiron = torch.nn.CrossEntropyLoss()

    preprocessor = ClassificationBasePreProcessor({"0": 0, "1": 1, "2": 2}, 128, tokenizer)
    train_set = NLPDataset(train_file, preprocessor, tokenizer)

    def collate_fn(batch):
        inputs = torch.tensor([x.inputs for x in batch], dtype=torch.int32)
        input_mask = torch.tensor([x.input_mask for x in batch], dtype=torch.int32)
        segment_ids = torch.tensor([x.segment_ids for x in batch], dtype=torch.int32)
        targets = torch.tensor([x.target for x in batch])

        return inputs, input_mask, segment_ids, targets#.unsqueeze(1)

    train_loader = DataLoader(train_set, 32, True, collate_fn=collate_fn)

    epoch = 20
    loss_list = []
    for epoch in tqdm.tqdm(range(epoch)):
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


