import re, collections

def get_stats(dictionary):
    # 유니그램의 pair들의 빈도수를 카운트
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    print('현재 pair들의 빈도수 :', dict(pairs))
    return pairs

def merge_dictionary(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

if __name__ == "__main__":
    num_merges = 10
    dictionary = {'l o w </w>' : 5,
             'l o w e r </w>' : 2,
             'n e w e s t </w>':6,
             'w i d e s t </w>':3
             }

    bpe_codes = {}
    bpe_codes_reverse = {}

    for i in range(num_merges):
        print("### Iteration {}".format(i + 1))
        pairs = get_stats(dictionary)
        best = max(pairs, key=pairs.get)
        dictionary = merge_dictionary(best, dictionary)

    bpe_codes[best] = i
    bpe_codes_reverse[best[0] + best[1]] = best

    print("new merge: {}".format(best))
    print("dictionary: {}".format(dictionary))
