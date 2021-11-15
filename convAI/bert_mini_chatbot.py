import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import os
from dataclasses import dataclass
import torch
from torch import nn

class Config:
    def __init__(self, chatbot_data=None, para_kqc_data = None,
                 questions = None, bert_multilingual = None):
        if chatbot_data is None:
            self.ChatbotData = "../data/chatbot_data.txt"
        else:
            self.ChatbotData = chatbot_data

        if para_kqc_data is None:
            self.ParaKQCData = para_kqc_data
        else:
            self.ParaKQCData = "../data/paraKQC_v1.txt"

        if questions is None:
            self.Questions = questions
        else:
            self.Questions = "../data/questions_embeddings.npy"

        if bert_multilingual is None:
            self.BERTMultiLingual = bert_multilingual
        else:
            self.BERTMultiLingual = "bert-base-multilingual-cased"

        self.Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Similarity:
    def __init__(self):
        print("Similarity")

    def get_l2_distance(self, x1, x2, dim: int = 1):
        return ((x1 - x2) ** 2).sum(dim=dim) ** .5

    def get_l1_distance(self, x1, x2, dim: int = 1):
        return ((x1 - x2).abs()).sum(dim=dim)

    def get_cosine_similarity(self, x1, x2, dim: int = 1):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(x1, x2)


class VanillaChatbot:
    def __init__(self, config=None, sim_type: str = 'cos'):
        self.config = config
        print(self.config.BERTMultiLingual)
        self.model = AutoModel.from_pretrained(self.config.BERTMultiLingual)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.BERTMultiLingual)
        self.questions = self.load_questions(self.config.Questions)
        self.answers = pd.read_csv(self.config.ChatbotData)["A"].tolist()
        self.measure = self.get_similarity_measure(sim_type=sim_type)  # 유사도 측정 객체

    def query(self, question: str, return_answer=False):
        question_tokenized = self.tokenizer(question, return_tensors="pt")
        question_embedded = self.model(**question_tokenized).pooler_output
        similar_question_id = self.get_similar_question_id(question_embedded)
        answer = self.answers[similar_question_id]
        print('챗봇:', answer)
        if return_answer:
            return answer

    def get_similar_question_id(self, q_emb):
        similarities = self.measure(self.questions, q_emb)
        similar_question_id = torch.argmax(similarities).item()
        return similar_question_id

    @staticmethod
    def load_questions(root):
        questions = torch.tensor(np.load(root))
        return questions

    @staticmethod
    def get_similarity_measure(sim_type: str):
        if sim_type == 'cos':
            measure = Similarity().get_cosine_similarity
        elif sim_type == 'l1':
            measure = Similarity().get_l1_distance
        elif sim_type == 'l2':
            measure = Similarity().get_l2_distance

        return measure

