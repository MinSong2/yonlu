
from yonlu.word_embeddings.word_embeddings import Word2Vec

word2vec = Word2Vec()
mode = 'unfiltered'
mecab_path = 'C:\\mecab\\mecab-ko-dic'
stopword_file = '../stopwords/stopwordsKor.txt'
files = []
files.append('../data/content.txt')
is_directory=False
doc_index=-1
max=-1
is_mecab=False
word2vec.preprocessing(mode,is_mecab,mecab_path,stopword_file,files,is_directory,doc_index,max)

min_count=1
window=5
size=200
negative=0
word2vec.train(min_count, window, size, negative)

model_file = 'word2vec_1.bin'
binary=True;
word2vec.save_model(model_file, binary)


