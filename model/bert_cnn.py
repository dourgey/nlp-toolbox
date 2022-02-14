import tqdm
from transformers import BertModel, BertConfig, BertTokenizerFast
import torch
import torch.nn as nn
from .model_config import ModelConfig


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

