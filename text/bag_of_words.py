


def make_bag_of_words(corpus:'sequence of strings') -> ("dictionary","matrix"):
    import re
    if not (type(corpus) in (list,tuple) and all(type(e) in (str,bytes) for e in corpus)):
        raise TypeError("bad argument")

    pattern = re.compile(r'\b[a-z]{3,15}\b', re.IGNORECASE) # excludes digits and short words
    pattern = re.compile(r'\b\w\w+\b', re.IGNORECASE)       # excudes digits, words at least two characters long

    ll = [re.findall(pattern, (e if isinstance(e,str) else str(e, encoding='utf-8')).lower().strip())
                            for e in corpus]    # ll = tokenized/lemmitized corpus
    d = {s:j for j,s in enumerate(sorted(set(sum(ll, []))))} # d = vocabulary

    #the one approach
    mx = [[None,]*len(d.keys()) for e in corpus]
    for i,l in enumerate(ll):
        for s,j in d.items():
            mx[i][j] =  l.count(s)

    #a different approach (slower in terms of time)
    from collections import Counter
    mx = [[0,]*len(d.keys()) for e in corpus]
    for i,l in enumerate(ll):
        counter = Counter(l)
        [mx[i].__setitem__(d[word], count) for word,count in counter.items()]
    return (d,mx)
#==================================================================================

def main():
    with open("text.txt", mode='wb') as fh:
        bt = b'document #3: document loaded from this file'
        fh.write(bt)
    document3 = open("text.txt").read()

    corpus = ["document #1: make a bag-of-words out of this corpus",  #document1
              "document #2: discard punctuation from this document!", #document2
              document3]                                              #document3
    target = (0,1,1)
    d,mx = make_bag_of_words(corpus)
    print(d)
    print(mx.__str__().replace(r"], [",  '],\n ['))
if __name__=='__main__':main()


















