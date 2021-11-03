import json
import pickle
import torch
from gluonnlp.data import SentencepieceTokenizer
from yonlu.model.net import KobertCRF
from yonlu.data_utils.utils import Config
from yonlu.data_utils.vocab_tokenizer import Tokenizer
from yonlu.data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

from yonlu.ner.predict_bert_crf import DecoderFromNamedEntitySequence


def main():
    mode = 'bert_only'
    if mode == 'bert_only':
        model_dir = '../experiments/base_model'
    elif mode == 'bert_crf':
        model_dir = '../experiments/base_model_with_crf'
    elif mode == 'bert_bilstm_crf':
        model_dir = '../experiments/base_model_with_bilstm_crf'
    elif mode == 'bert_bigru_crf':
        model_dir = '../experiments/base_model_with_bigru_crf'

    model_dir = Path(model_dir)
    model_config = Config(json_path=str(model_dir) + '/config.json')

    # load vocab & tokenizer
    tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    with open(str(model_dir) + "/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(str(model_dir) + "/ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    checkpoint = torch.load("../experiments/base_model_with_crf/best-epoch-9-step-750-acc-0.960.bin", map_location=torch.device('cpu'))

    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    input_text = "지난달 28일 수원에 살고 있는 윤주성 연구원은 코엑스(서울 삼성역)에서 개최되는 DEVIEW 2019 Day1에 참석했다. LaRva팀의 '엄~청 큰 언어 모델 공장 가동기!' 세션을 들으며 언어모델을 학습시킬때 multi-GPU, TPU 모두 써보고 싶다는 생각을 했다."
    list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
    x_input = torch.tensor(list_of_input_ids).long()
    list_of_pred_ids = model(x_input)

    list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids,
                                                               list_of_pred_ids=list_of_pred_ids)
    print("output>", decoding_ner_sentence)
    print("")

if __name__ == '__main__':
    main()
