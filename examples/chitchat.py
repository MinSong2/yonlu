
from yonlu.convAI import bert_mini_chatbot
import fire

from yonlu.convAI.bert_mini_chatbot import Config
from yonlu.convAI.export_question_embeddings import export_question_embeddings
from os.path import exists

EXIT = "대화종료"

if __name__ == "__main__":
    chatbotData = "../data/ChatbotData.csv"
    paraKQCData = "../data/paraKQC_v1.txt"
    questions = "../data/questions_embeddings.npy"
    BERTMultiLingual = "bert-base-multilingual-cased"
    config = Config(chatbot_data=chatbotData, para_kqc_data=paraKQCData,
                    questions=questions, bert_multilingual=BERTMultiLingual)

    if exists(questions) != True:
        fire.Fire({"run": export_question_embeddings(config)})

    print('*' * 150)
    print('환영합니다! 간단한 챗봇을 사용해보세요')
    print('*' * 150)


    bot = bert_mini_chatbot.VanillaChatbot(config)
    while True:
        text = input("할 말을 입력해주세요(종료시 '대화종료' 입력): ")
        if text == EXIT:
            break
        bot.query(text)
    print('*' * 150)
    print('챗봇을 종료합니다. 다음에 또 만나요!')
    print('*' * 150)