from utils.data import NLPDataset, ClassificationBasePreProcessor

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast("../distilbert/vocab.txt")
config = DistilBertConfig.from_pretrained("../distilbert")
distilbert_model = DistilBertModel(config).from_pretrained("../distilbert")

print(distilbert_model)

train_file = "ertai_train.json"

preprocessor = ClassificationBasePreProcessor({"0": 0, "1": 1, "2": 2}, 256, tokenizer)
ds = NLPDataset(train_file, preprocessor, tokenizer)
print(ds[1])