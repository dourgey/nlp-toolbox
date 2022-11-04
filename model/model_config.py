import os
from utils.json_utils import *
from abc import abstractmethod
import toml

class ModelConfig():
    @abstractmethod
    def __init__(self, config_file=None, **kwargs):
        self.config_file = config_file

    def save(self, file: str) -> None:
        with open(file, 'w', encoding='utf-8') as f:
            toml.dump(self.__dict__, f)


    @classmethod
    def from_config_file(cls, config_file):
        assert os.path.exists(config_file)
        with open(config_file, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        config = cls(**config)
        return config

    def __str__(self):
        text = f"{'██' * 10} {self.__class__.__name__} {'██' * 10} \n"
        for k in self.__dict__:
            text += f"▨\t{k}: {self.__dict__[k]}\n"
        text += '█' * (len(f"{'==' * 10} {self.__class__.__name__} {'==' * 10} \n") - 2)
        return text

