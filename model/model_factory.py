import os

from model.base_model import MODEL_TYPE, MODEL_NAME, BaseModel
from model.model_config import ModelConfig
from model.textcnn import TextCNNConfig, TextCNN


class CreateFactory:
    def create_model(self, config: ModelConfig, model_type: MODEL_TYPE, model_name: MODEL_NAME) -> BaseModel:
        assert isinstance(model_type, MODEL_TYPE)
        assert isinstance(model_name, MODEL_NAME)

        if model_type == MODEL_TYPE.TEXT_CLASSIFICATION:
            assert model_name in [MODEL_NAME.TEXTCNN, MODEL_NAME.RNN, MODEL_NAME.TRANSFORMERS, MODEL_NAME.BERT]

            if model_name == MODEL_NAME.TEXTCNN:
                assert isinstance(config, TextCNNConfig)
                model = TextCNN(config)


        return model


    def create_config(self, config_file, model_type: MODEL_TYPE, model_name: MODEL_NAME):
        assert isinstance(model_type, MODEL_TYPE)
        assert isinstance(model_name, MODEL_NAME)

        if model_type == MODEL_TYPE.TEXT_CLASSIFICATION:
            assert model_name in [MODEL_NAME.TEXTCNN, MODEL_NAME.RNN, MODEL_NAME.TRANSFORMERS, MODEL_NAME.BERT]

            if model_name == MODEL_NAME.TEXTCNN:
                assert os.path.exists(config_file)
                config = TextCNNConfig.from_config_file(config_file=config_file)


        return config



if __name__ == '__main__':
    factory = CreateFactory()
    config_file = '../common/configs/textcnn.conf.toml'
    config = factory.create_config(config_file, MODEL_TYPE.TEXT_CLASSIFICATION, MODEL_NAME.TEXTCNN)
    model = factory.create_model(config, MODEL_TYPE.TEXT_CLASSIFICATION, MODEL_NAME.TEXTCNN)
    print(model)
