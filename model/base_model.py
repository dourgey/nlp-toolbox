import torch.nn as nn
from enum import Enum

class MODEL_TYPE(Enum):
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_LABELING = "sequence_labeling"
    SIMILARITY = "similarity"

class MODEL_NAME(Enum):
    TEXTCNN = "textcnn"
    RNN = "rnn"
    TRANSFORMERS = "transformers"
    BERT = "bert"
    BERT_CNN = "bert_cnn"


class BaseModel(nn.Module):
    pass

class ModelForTextClassification(BaseModel):
    pass


class ModelForSequenceLabeling(BaseModel):
    pass


class ModelForSimilarity(BaseModel):
    pass

