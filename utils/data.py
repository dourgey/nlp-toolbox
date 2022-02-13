import os

import torch
from torch.utils.data import DataLoader, Dataset
from utils.json_utils import json_to_obj
from abc import ABCMeta, abstractmethod
from types import SimpleNamespace
from typing import List, Dict
from utils.my_log import logger


class DataEntity:
    def __init__(self, inputs=None, target=None, **kwargs):
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


class IPreProcessor(metaclass=ABCMeta):
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
    基础文本分类Collator
    由于要兼容之前的格式，注意对之前格式进行处理，例如__call__()函数的前两行
    """
    def __init__(self, label_list, max_seq_length, tokenizer):
        """

        :param label_list: dict of label discription like {}
        :param max_seq_length:
        :param tokenizer:
        """
        super(ClassificationBasePreProcessor, self).__init__()
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __call__(self, data: List[SimpleNamespace]) -> List[DataEntity]:
        data_entities = []
        for example in data:
            inputs = self.tokenizer.tokenize(example.inputs)
            if len(inputs) > self.max_seq_length - 2:
                inputs = inputs[: self.max_seq_length - 2]
            tokens = ["[CLS]"] + inputs + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            target = 0 if "predict_label" == example.labels else self.label_list[example.labels]
            data_entities.append(
                DataEntity(inputs=input_ids, target=target, input_mask=input_mask, segment_ids=segment_ids))
        return data_entities

    @staticmethod
    def collator(batch_data):
        pass




class NLPDataset(Dataset):
    def __init__(self, input_file, preprocessor: IPreProcessor, tokenizer):
        """
        用于nlp-toolbox的数据集类

        :param input_file: 输入文件，JSON格式 / 旧格式自动识别
        :param preprocessor: IPreProcessor子类，负责数据预处理
        :param batch_size: 批大小
        :param tokenizer: tokenizer, convert token to id
        :param re_convert_tfrecord: 是否转换为tfrecord格式
        :param out_file: tfrecord格式存储文件，re_covert_tfrecord=True时，必须指定该路径
        :param shuffle_buffer_size:
        """
        self.is_json = True
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

    @classmethod
    def _read_txt(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as fp:
            line_list = []
            for line in fp:
                tokens = line.strip().split("[SEP]")
                line_list.append(tokens)
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
    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=r"D:\Works\codes\wubanlp_dev\data\demo\bert_vocab.txt", do_lower_case=False)
    # rc = ClassificationWithRoleCollator(label_list=['__label__right', '__label__wrong'], name_to_features=name_to_features, tokenizer=tokenizer, max_seq_length=512)

    # for (input_ids, role_ids, input_mask, segment_ids, _), target in dataset:
    #     print("inputs:", input_ids,
    #            "\nrole_ids:", role_ids,
    #            "\ninput_mask:", input_mask,
    #            "\nsegment_ids:", segment_ids,
    #            "\ntarget:", _,
    #           "\ntarget:", target)

    txt_file = r"C:\Users\Zed\Downloads\test.txt"
    json_file = r"D:\Works\codes\wubanlp_dev\data\demo\classification\test.json"
    #
    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=r"D:\Works\codes\wubanlp_dev\data\demo\bert_vocab.txt", do_lower_case=False)
    # rc = ClassificationBaseCollator(label_list=['__label__right', '__label__wrong'], tokenizer=tokenizer, max_seq_length=512)
    # dataset = Dataset(json_file, rc, 32, True, False, out_file=r"D:\Works\codes\wubanlp_dev\data\demo\classification\test.tfrecord")
    #
    # for d in dataset:
    #     (input_ids, input_mask, segment_ids, _), target = d
    #     print("inputs:", input_ids,
    #            "\ninput_mask:", input_mask,
    #            "\nsegment_ids:", segment_ids,
    #            "\ntarget:", _,
    #           "\ntarget:", target)
    #     break



    from transformers import BertTokenizer, BertModel, BertConfig
    bert_vocab = r'D:\Works\codes\voice_quality\voice_quality_labeling\voice-quality-labeling-srv\target\classes\config_online\label_config\hbgprivatecall\house\bert_vocab.txt'
    tokenizer = BertTokenizer(bert_vocab)
    print(tokenizer(["测试", "测试2"], padding=True))

    inputs = {'input_ids': torch.tensor([[101, 3844, 6407, 102, 0], [101, 3844, 6407, 123, 102]]), 'token_type_ids': torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])}
    config = BertConfig()
    bert = BertModel(config)
    print(bert(inputs))