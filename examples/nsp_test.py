from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
prompt = "The child came home from school."
next_sentence = "He played soccer after school."
encoding = tokenizer.encode_plus(prompt, next_sentence, return_tensors='pt')
outputs = model(**encoding)[0]
softmax = F.softmax(outputs, dim = 1)
print(softmax)