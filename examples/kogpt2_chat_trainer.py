
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from yonlu.convAI.gpt2_chitchat_trainer import KoGPT2Chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chitchat based on KoGPT-2')
    parser.add_argument('--chat',
                        action='store_true',
                        default=True,
                        help='response generation on given user input')

    parser.add_argument('--sentiment',
                        type=str,
                        default='0',
                        help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

    parser.add_argument('--model_params',
                        type=str,
                        default='checkpoints/epoch=2-step=371.ckpt',
                        help='model binary for starting chat')

    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='for training')

    parser.add_argument('--num_worker',
                        type=int,
                        default=0,
                        help='number of workers: in CPU, there might an issue across processes in parallelization. For that case, set num-worker to 0')

    parser.add_argument('--train_file',
                        type=str,
                        default='../data/ChatbotData.csv',
                        help='train file')


    parser = KoGPT2Chat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer(accelerator="cpu", gpus=0).from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0, logger=False)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()