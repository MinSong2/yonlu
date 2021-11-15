from tqdm import tqdm
import pandas as pd
import numpy as np
import fire
from transformers import AutoModel, AutoTokenizer
from .bert_mini_chatbot import Config
import torch

def export_question_embeddings(config=None, save_path: str = "../data/questions_embeddings.npy"):
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(config.BERTMultiLingual).cuda()
    else:
        model = AutoModel.from_pretrained(config.BERTMultiLingual)

    tokenizer = AutoTokenizer.from_pretrained(config.BERTMultiLingual)

    questions = pd.read_csv(config.ChatbotData)["Q"].tolist()
    if torch.cuda.is_available():
        questions = list(
           map(lambda x: tokenize_cuda(x, tokenizer), questions)
        )  # tokenize & allocate cuda
    else:
        questions = list(
            map(lambda x: tokenize(x, tokenizer), questions)
        )
    question_embeddings = []
    for q in tqdm(questions, desc="Getting embedding outputs"):
        question_embeddings.append(model(**q).pooler_output.cpu().detach().numpy())

    question_embeddings_agg = np.vstack(question_embeddings)

    # save phase
    with open(save_path, "wb") as f:
        np.save(f, question_embeddings_agg)

    print(f"Question embedding vectors saved in {save_path}😎")

    return


def tokenize_cuda(x, tokenizer):
    x = tokenizer(x, return_tensors="pt")
    x["input_ids"] = x["input_ids"].cuda()
    x["token_type_ids"] = x["token_type_ids"].cuda()
    x["attention_mask"] = x["attention_mask"].cuda()
    return x


def tokenize(x, tokenizer):
    x = tokenizer(x, return_tensors="pt")
    return x

if __name__ == "__main__":
    fire.Fire({"run": export_question_embeddings})