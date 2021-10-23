import torch
from transformers import BertTokenizer
import os

from yonlu.data_utils.translation_dataset import TranslationDataset
from yonlu.machine_translation.train_translation import TranslationTrainer
from yonlu.model.transformer import Transformer

if __name__ == '__main__':
  torch.manual_seed(10)
  dir_path = '../machine_translation'
  vocab_path = '../data/wiki-vocab.txt'
  data_path = '../data/nmt_ko_en.csv'
  #data_path = '../data/test.csv'
  checkpoint_path = '../checkpoints'

  if not os.path.exists(checkpoint_path):
      os.makedirs(checkpoint_path)

  # model setting
  model_name = 'transformer-translation'
  vocab_num = 22000
  max_length = 64
  d_model = 512
  head_num = 8
  dropout = 0.1
  N = 6
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

  # hyper parameter
  epochs = 50
  batch_size = 4
  padding_idx = tokenizer.pad_token_id
  learning_rate = 0.5

  dataset = TranslationDataset(tokenizer=tokenizer, file_path=data_path, max_length=max_length)

  model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N)

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

  trainer = TranslationTrainer(dataset, tokenizer, model, max_length, device, model_name, checkpoint_path, batch_size)
  train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.2)

  trainer.train(epochs, train_dataloader, eval_dataloader, optimizer, scheduler)

