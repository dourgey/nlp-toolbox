import os

import pandas as pd
from sklearn.metrics import classification_report

from common.module.trainer import *
from utils.json_utils import dict_to_json


class Evaluator:
    def __init__(self, model, model_type, label_list, device):
        self.model = model
        assert model_type in ['classification', 'labeling']
        self.model_type = model_type
        self.device = device
        self.label_list = label_list


    def _classification_evaluate(self, dataloader):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in dataloader:
                x = [_x.to(self.device) for _x in x]
                y_hat = torch.softmax(self.model(x), dim=-1).max(dim=-1)[1]
                y_pred += y_hat.cpu().numpy().tolist()
                y_true += y.numpy().tolist()
        print("====================== evaluation result ======================")
        print(classification_report(y_true, y_pred, target_names=self.label_list))
        print("===============================================================")
        return classification_report(y_true, y_pred, target_names=self.label_list, output_dict=True)

    def _classification_predict(self, dataset, collator_fn):
        predict_results = []
        self.model.eval()
        with torch.no_grad():
            for data in dataset:
                (x, y) = collator_fn([data])
                x = [_x.to(self.device) for _x in x]
                y_pred = torch.softmax(self.model(x), dim=-1)[0]
                predict_results.append({"context": data.context,
                                        "scores": {self.label_list[i]: y_pred[i].item() for i in range(len(self.label_list))}})
        return predict_results

    @staticmethod
    def out_to_file(output_dir, predict_results):
        with open(os.path.join(output_dir, 'predict_results.json'), 'w', encoding='utf-8') as f:
            f.write(dict_to_json(predict_results))


    def eval(self, dataloader, output_dir=None):
        if self.model_type == 'classification':
            eval_result = self._classification_evaluate(dataloader)
        if output_dir:
            pd.DataFrame(eval_result).transpose().to_csv(os.path.join(output_dir, 'classification_report.csv'))

    def predict(self, dataset, collate_fn, output_dir=None):
        if self.model_type == 'classification':
            predict_results = self._classification_predict(dataset, collate_fn)

        if self.model_type == 'labeling':
            pass

        # 将预测结果写入文件
        if output_dir:
            self.out_to_file(output_dir, predict_results)





if __name__ == '__main__':
    from transformers import BertTokenizer

    train_file = '../test/iPhone/train.json'
    tokenizer = BertTokenizer.from_pretrained('../../test/bert_www')
    preprocessor = ClassificationBasePreProcessor(label_list=["0", "1", "2"], max_seq_length=128, tokenizer=tokenizer)

    train_set = NLPDataset('../../test/iPhone/train.json', preprocessor, tokenizer, is_predict=False)
    valid_set = NLPDataset('../../test/iPhone/dev.json', preprocessor, tokenizer, is_predict=False)
    test_set = NLPDataset('../../test/iPhone/test.json', preprocessor, tokenizer, is_predict=True)
    print(len(test_set))
    train_loader = DataLoader(train_set, batch_size=64, collate_fn=NLPCollator.classificationBaseCollateFn, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=64, collate_fn=NLPCollator.classificationBaseCollateFn, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=NLPCollator.classificationBaseCollateFn, shuffle=False)
    config = TextCNNConfig.from_config_file('../configs/textcnn.conf')
    print(config)
    mm = TextCNN(config)
    trainer = Trainer(train_loader, valid_loader, mm, optimizer=torch.optim.Adam(mm.parameters(), lr=1e-4), criterion=nn.CrossEntropyLoss(), test_loader=None, ckpt_path='../../test/ckpt/textcnn.ckpt', device='cuda', early_stop=True)
    trainer.train(epoch_num=100)

    # predict(model, test_set, NLPCollator.classificationBaseCollateFn, ['0', '1', '2'], 'cuda', output_dir='./')
    # eval(model, test_loader, ['0', '1', '2'], 'cuda', task_name='classification', output_dir='./')
    from torchinfo import summary
    summary(mm)

    evaluator = Evaluator(mm, 'classification', ['0', '1', '2'], 'cuda')
    evaluator.eval(test_loader, output_dir='./')
    evaluator.predict(test_set, NLPCollator.classificationBaseCollateFn, output_dir='./')