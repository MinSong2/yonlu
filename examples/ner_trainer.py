
from yonlu.ner.train_bert_crf import train

if __name__ == '__main__':
    model_dir = '../experiments/base_model_with_crf'
    train_data_dir = "../data_in/NER-master/말뭉치 - 형태소_개체명"
    val_data_dir = "../data_in/NER-master/validation_set"
    train(model_dir, train_data_dir, val_data_dir)