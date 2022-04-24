import os

import torch
from torch.utils.data import DataLoader, Dataset
from utils.json_utils import json_to_obj
from abc import ABCMeta, abstractmethod
from types import SimpleNamespace
from typing import List, Dict
from utils.my_log import logger


class DataEntity:
    def __init__(self, inputs, target, **kwargs):
        """
        用于nlp-toolbox的数据实体定义，

        :param inputs:  obj，模型所需的输入, such as text
        :param target:  obj，模型的学习目标，such as label etc
        :param kwargs:  obj， optional 扩展字段, 约定字段： 'sentencePosition', 'rolePosition'
            'sentencePosition' 表示文本的句子编号信息， 'rolePosition' 表示文本的角色信息
        """
        self.inputs = inputs
        self.target = target
        for key in kwargs:
            self.__dict__[key] = kwargs[key]


    def to(self, device):
        for key in self.__dict__:
            self[key].to(device)


class IPreProcessor(metaclass=ABCMeta):
    """
    数据预处理基类，用于将原始数据转换为DataEntity
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass


    @abstractmethod
    def __call__(self, data: List[SimpleNamespace]) -> Dict:
        """
                处理解析JSON文件后生成的数据
                :param data: 解析JSON得到的数据
                :return: DataEntity列表，每条数据处理后为一个DataEntity
                """
        pass


class ClassificationBasePreProcessor(IPreProcessor):
    """
    基础文本分类PreProcessor
    由于要兼容之前的格式，注意对之前格式进行处理，例如__call__()函数的前两行
    """
    def __init__(self, label_list, max_seq_length, tokenizer):
        """
        PreProcessor初始化
        :param label_list: dict of label discription like {}
        :param max_seq_length:
        :param tokenizer:
        """
        super(ClassificationBasePreProcessor, self).__init__()
        self.label_list = {label_list[i]: i for i in range(len(label_list))}
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __call__(self, data: List[SimpleNamespace]) -> List[DataEntity]:
        """
        将解析JSON得到的数据转换为DataEntity列表，按Berts的输入格式进行处理（inputs, input_mask, segment_ids）
        :param data: 解析JSON得到的数据
        :return: DataEntity列表，每条数据处理后为一个DataEntity
        """
        data_entities = []
        for example in data:
            inputs = self.tokenizer.tokenize(example.inputs)
            if len(inputs) > self.max_seq_length - 2:
                inputs = inputs[: self.max_seq_length - 2]
            tokens = ["[CLS]"] + inputs + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            target = 0 if "predict_label" == example.labels else self.label_list[example.labels]
            data_entities.append(
                DataEntity(inputs=input_ids, target=target, input_mask=input_mask, segment_ids=segment_ids))
        return data_entities


class NLPCollator:
    """
    各个任务的collator_fn, 用于DataLoader的collate_fn，将DataEntity列表按需转换为torch.Tensor
    """
    @staticmethod
    def classificationBaseCollateFn(batch):
        """
        基础文本分类的collate_fn
        :param batch: list of DataEntity
        :return: tuple: (inputs, input_mask, segment_ids), target
        """
        maxlen = max([len(x.inputs) for x in batch])
        inputs = []
        input_mask = []
        segment_ids = []
        y = []
        for x in batch:
            inputs.append(torch.tensor(x.inputs + [0] * (maxlen - len(x.inputs))))
            input_mask.append(torch.tensor(x.input_mask + [0] * (maxlen - len(x.input_mask))))
            segment_ids.append(torch.tensor(x.segment_ids + [0] * (maxlen - len(x.segment_ids))))
            y.append(x.target)
        return (torch.stack(inputs), torch.stack(input_mask), torch.stack(segment_ids)), torch.tensor(y)

    @staticmethod
    def LabelingBaseCollateFn(batch):
        pass


class NLPDataset(Dataset):
    def __init__(self, input_file, preprocessor: IPreProcessor, tokenizer):
        """
        用于nlp-toolbox的数据集类

        :param input_file: 输入文件，JSON格式
        :param preprocessor: IPreProcessor子类，负责数据预处理
        :param batch_size: 批大小
        :param tokenizer: tokenizer, convert token to id
        """
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        data = self._read_json(input_file)
        self.data = preprocessor(data)


    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as fp:
            line_list = []
            for line in fp:
                line_list.append(json_to_obj(line))
            return line_list


    def __iter__(self):
        return self.__next__()

    def __next__(self):
        for x in self.data:
            yield x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]





if __name__ == '__main__':
    from transformers import BertTokenizer
    train_file = '../test/iPhone/train.json'
    tokenizer = BertTokenizer.from_pretrained('../test/distilbert')
    preprocessor = ClassificationBasePreProcessor(label_list=["0", "1", "2"], max_seq_length=128, tokenizer=tokenizer)

    train_set = NLPDataset(train_file, preprocessor, tokenizer)
    dataloader = DataLoader(train_set, batch_size=2, collate_fn=NLPCollator.classificationBaseCollateFn)

    for X, y in dataloader:
        print(X, y)
        break
