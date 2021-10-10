from __future__ import absolute_import, division, print_function, unicode_literals

class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append(
                        {"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-" + entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": entity_tag, "prob": None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""

                # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
            decoding_ner_sentence = ""
            is_prev_entity = False
            prev_entity_tag = ""
            is_there_B_before_I = False

            for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
                if i == 0 or i == len(pred_ner_tag) - 1:  # remove [CLS], [SEP]
                    continue
                token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체
                if 'B-' in pred_ner_tag_str:
                    if is_prev_entity is True:
                        decoding_ner_sentence += ':' + prev_entity_tag + '>'

                    if token_str[0] == ' ':
                        token_str = list(token_str)
                        token_str[0] = ' <'
                        token_str = ''.join(token_str)
                        decoding_ner_sentence += token_str
                    else:
                        decoding_ner_sentence += '<' + token_str
                    is_prev_entity = True
                    prev_entity_tag = pred_ner_tag_str[-3:]  # 첫번째 예측을 기준으로 하겠음
                    is_there_B_before_I = True

                elif 'I-' in pred_ner_tag_str:
                    decoding_ner_sentence += token_str

                    if is_there_B_before_I is True:  # I가 나오기전에 B가 있어야하도록 체크
                        is_prev_entity = True
                else:
                    if is_prev_entity is True:
                        decoding_ner_sentence += ':' + prev_entity_tag + '>' + token_str
                        is_prev_entity = False
                        is_there_B_before_I = False
                    else:
                        decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence

