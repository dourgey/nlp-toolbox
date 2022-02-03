import os
from utils.json_utils import *
from abc import ABCMeta, abstractmethod

class ModelConfig():
    @abstractmethod
    def __init__(self, config_file=None, **kwargs):
        self.config_file = config_file

    def save(self, file: str) -> None:
        open(file, 'w').write(dict_to_json(self.__dict__))

    @classmethod
    def from_config_file(cls, config_file):
        assert os.path.exists(config_file)
        config = cls(**json_to_dict(open(config_file, 'r').read()))
        return config

    def __str__(self):
        text = f"{'██' * 10} {self.__class__.__name__} {'██' * 10} \n"
        for k in self.__dict__:
            text += f"▨\t{k}: {self.__dict__[k]}\n"
        text += '█' * (len(f"{'==' * 10} {self.__class__.__name__} {'==' * 10} \n") - 2)
        return text

