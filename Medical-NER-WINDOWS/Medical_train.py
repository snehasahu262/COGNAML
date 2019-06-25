from Medicalmodel.Medical_data_utils import CoNLLDataset
from Medicalmodel.Medical_ner_model import NERModel
from Medicalmodel.config import Config


def main():
   
    config = Config()

    model = NERModel(config)
    model.build()
    
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

   
    model.train(train, dev)

if __name__ == "__main__":
    main()
