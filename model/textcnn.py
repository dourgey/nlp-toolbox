import torch
import torch.nn as nn

from typing import List
from model.model_config import ModelConfig


class TextCNNConfig(ModelConfig):
    def __init__(self, num_classes, n_vocab, embedding_dim, num_filters, filter_sizes: List[int],
                 dilation_rate: List[int], embedding_pretrained=None, dropout_prob=0.1, config_file=None):
        """
        TextCNN 模型参数，如需读取配置文件，请使用TextCNNConfig.from_config_file(config_path)

        :param num_classes: 分为多少类
        :param n_vocab: 词汇表大小
        :param embedding_dim: 词嵌入维度
        :param num_filters: 卷积核数量
        :param filter_sizes: 卷积核尺寸
        :param dilation_rate: 卷积核空洞率
        :param embedding_pretrained:
        :param dropout_prob: dropout概率
        :param config_file: config文件路径
        """
        super(TextCNNConfig, self).__init__(config_file)
        self.embedding_dim = embedding_dim
        self.n_vocab = n_vocab
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dilation_rate = dilation_rate
        self.embedding_pretrained = embedding_pretrained
        self.dropout_prob = dropout_prob


class TextCNN(nn.Module):
    def __init__(self, config: TextCNNConfig):
        super(TextCNN, self).__init__()
        self.config = config

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embedding_dim))
                                    for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout_prob)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x)).squeeze(3)
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x[0])
        x = x.unsqueeze(1)
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


if __name__ == '__main__':
    config = TextCNNConfig.from_config_file(config_file='../common/configs/textcnn.conf')
    print(config)