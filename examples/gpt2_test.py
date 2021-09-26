import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

sequence = "Worldwide, numerous memorials have been dedicated to Churchill. His statue in Parliament Square was unveiled by his widow Clementine in 1973 and is one of only twelve in the square, all of prominent political figures, including Churchill's friend Lloyd George and his India policy nemesis Gandhi.[459][460] Elsewhere in London, the wartime Cabinet War Rooms have been renamed the Churchill Museum and Cabinet War Rooms.[461] Churchill College, Cambridge, was established as a national memorial to Churchill. An indication of Churchill's high esteem in the UK is the result of the 2002 BBC poll, attracting 447,423 votes, in which he was voted the greatest Briton of all time, his nearest rival being Isambard Kingdom Brunel some 56,000 votes behind.[462] He is one of only eight people to be granted honorary citizenship of the United States; others include Lafayette, Raoul Wallenberg and Mother Teresa.[463] The United States Navy honoured him in 1999 by naming a new Arleigh Burke-class destroyer as the USS Winston S. Crchill.[464] Other memorials in North America include the National Churchill Museum in Fulton, Missouri, where he made the 1946 \"Iron Curtain\" speech; Churchill Square in central Edmonton, Alberta; and the Winston Churchill Range, a mountain range northwest of Lake Louise, also in Alberta, which was renamed after Churchill in 1956.[465]"

#Because we are using PyTorch, we add return_tensor='pt',
# if using TensorFlow, we would use return_tensor='tf'.
inputs = tokenizer.encode(sequence, return_tensors='pt')

# To generate text with GPT-2, all we do is call the model.generate method
# we pass a maximum output length of 200 tokens
outputs = model.generate(inputs, max_length=200, do_sample=True)

# Our generate step outputs an array of tokens rather than words.
# To convert these tokens into words, we need to .decode them
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)