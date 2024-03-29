import json
import sys
from types import SimpleNamespace
from typing import Any
sys.path.append("../")

import toml

# author: shadowpower
# github: https://github.com/ShadowPower

def json_to_dict(json_str: str) -> dict:
    """
    JSON 转字典
    :param json_str: JSON
    :return: 字典
    """
    return json.loads(json_str)


def dict_to_json(obj: dict or list) -> str:
    """
    字典转 JSON
    :param obj: 字典
    :return: JSON
    """
    return json.dumps(obj, ensure_ascii=False, indent=4)


def json_to_obj(json_str: str) -> SimpleNamespace:
    """
    JSON 转对象（SimpleNamespace）
    :param json_str: JSON
    :return: 对象
    """
    return json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))


def obj_to_json(obj: Any) -> str:
    """
    对象转 JSON
    :param obj: 对象
    :return: JSON
    """
    return json.dumps(obj, default=lambda o: o.__dict__, ensure_ascii=False)


if __name__ == '__main__':
    with open('../common/configs/textcnn.conf.toml', 'r', encoding='utf-8') as f:
        s = f.read()
        print(toml.loads(s))