from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en')
fpath = "../data/2011_21.txt"
with open(fpath, "r", encoding='utf-8') as f:
    text = f.read()
    results = splitter.split(text=text)

for result in results:
    print(result)
