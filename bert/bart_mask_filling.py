from transformers import BartForConditionalGeneration, BartTokenizer

class BartMaskFilling:
    def __init__(self, model='facebook/bart-large', tokenizer='facebook/bart-large'):
        self.model = BartForConditionalGeneration.from_pretrained(model)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)

    def get_tokenizer(self):
        return self.tokenizer

    def predict(self, text, topk=5):
        input_ids = self.tokenizer([text], return_tensors='pt')['input_ids']
        logits = self.model(input_ids).logits
        masked_index = (input_ids[0] == self.tokenizer.mask_token_id).nonzero().item()
        probs = logits[0, masked_index].softmax(dim=0)
        values, predictions = probs.topk(topk)
        return self.tokenizer.decode(predictions)


if __name__ == "__main__":
    mask_filler = BartMaskFilling()
    text = "My friends are " + mask_filler.get_tokenizer().mask_token + " but they eat too many carbs."

    predicts = mask_filler.predict(text)
    for word in predicts.split():
        new_sentence = text.replace(mask_filler.get_tokenizer().mask_token, word)
        print(new_sentence)
