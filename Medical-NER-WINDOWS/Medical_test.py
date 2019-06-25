from Medicalmodel.Medical_data_utils import CoNLLDataset
from Medicalmodel.Medical_ner_model import NERModel
from Medicalmodel.config import Config
import pandas as pd


def align_data(data):
    
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()
    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
   
    '''
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> Fycompa is a tablet""")
    '''

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})
        list1=[]
        list2=[]
        for each1 in words_raw:
            list1.append(each1)
        for each2 in preds:
            list2.append(each2)
        res=dict(zip(list1,list2))
        res2=pd.DataFrame(res.items(),columns=['entity','value'])
        print(res2)
        csv_res=res2.to_csv('data_output.csv',sep='\t',encoding='utf-8')
       
        


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
