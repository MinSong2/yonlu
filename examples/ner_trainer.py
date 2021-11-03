
from yonlu.ner.train_bert_crf import train

if __name__ == '__main__':
    mode = 'bert_only'
    if mode == 'bert_only':
        model_dir = '../experiments/base_model'
    elif mode == 'bert_crf':
        model_dir = '../experiments/base_model_with_crf'
    elif mode == 'bert_bilstm_crf':
        model_dir = '../experiments/base_model_with_bilstm_crf'
    elif mode == 'bert_bigru_crf':
        model_dir = '../experiments/base_model_with_bigru_crf'

    train_data_dir = "../data_in/NER-master/말뭉치 - 형태소_개체명"
    val_data_dir = "../data_in/NER-master/validation_set"
    train(model_dir, train_data_dir, val_data_dir)