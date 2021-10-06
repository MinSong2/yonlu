import tensorflow as tf
from gensim.corpora import WikiCorpus
import os
import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def store(corpus, lang):
    base_path = os.getcwd()
    store_path = os.path.join(base_path, '{}_corpus'.format(lang))
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    file_idx=1
    for text in corpus.get_texts():
        current_file_path = os.path.join(store_path, 'article_{}.txt'.format(file_idx))
        with open(current_file_path, 'w' , encoding='utf-8') as file:
            file.write(bytes(' '.join(text), 'utf-8').decode('utf-8'))
        #endwith
        file_idx += 1
    #endfor

def tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list:
    return [token for token in text.split() if token_min_len <= len(token) <= token_max_len]

def run(lang):
    origin='https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2'.format(lang,lang)
    fname='{}wiki-latest-pages-articles.xml.bz2'.format(lang)
    file_path = tf.keras.utils.get_file(origin=origin, fname=fname, untar=False, extract=False)
    corpus = WikiCorpus(file_path, lemmatize=False, lower=False, tokenizer_func=tokenizer_func)
    store(corpus, lang)

if __name__ == '__main__':
    ARGS_PARSER = argparse.ArgumentParser()
    ARGS_PARSER.add_argument(
        '--lang',
        default='ko',
        type=str,
        help='language code to download from wikipedia corpus'
    )
    ARGS = ARGS_PARSER.parse_args()
    run(**vars(ARGS))