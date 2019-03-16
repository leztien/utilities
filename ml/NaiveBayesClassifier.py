

import numpy as np, pandas  as pd

def play_golf_table():
    from pandas import DataFrame, Series, Categorical, concat
    s,o,r = ['sunny','overcast','rainy']
    ct = Categorical([s,s,o,r,r,r,o,s,s,r,s,o,o,r])
    sr0 = Series(ct)
    h,m,c = ['hot','mild','cool']
    ct = Categorical([h,h,h,m,c,c,c,m,c,m,m,m,h,m])
    sr1 = Series(ct)
    h,n = "high normal".split()
    ct = Categorical([h,h,h,h,n,n,n,h,n,n,n,h,n,h])
    sr2 = Series(ct)
    sr3 = Series([int(s) for s in "01000110001101"])
    sr4 = Series([int(s) for s in "00111010111110"])
    nx = ['outlook','temperature','humidity','wind','play']
    df = DataFrame([sr0,sr1,sr2,sr3, sr4], index = nx).T
    return df

def df_into_mx(df, decoder=None):   #returns (integer-filled matrix, decoder as namedtuple)
    from numpy import array, zeros
    from pandas import factorize
    from collections import namedtuple
    factory = namedtuple("incoder", df.columns.values)
    l = []
    mx = zeros(shape=df.shape, dtype='i')
    for i,s in enumerate(df):
        sr = df[s]
        if 'categor' in str(type(sr)).lower(): sr = sr.astype(str)
        t = factorize(sr)
        mx[:,i] = t[0] if not hasattr(decoder, "__iter__") else [list(decoder[i]).index(s) for s in df.iloc[:,i]]
        l.append(t[-1])
    nt = factory(*l)
    return (mx, decoder or nt)


class  NaiveBayesClassifier():
    def __init__(self, X,y):
        from itertools import product
        from numpy import hstack
        self._l = []; yy = y.ravel(); self._y = y
        for xx in X.T:
            g = product(*[sorted(set(e)) for e in (xx, yy)])
            d = {}
            for t in g:
                nd = hstack([xx[:,None], yy[:,None]])
                p = len(xx[xx==t[0]]) / len(xx)     #marginal prob
                d.setdefault(t[0], p)
                nd = nd[nd[:,-1]==t[-1]]
                denominator = len(nd[nd[:,-1]==t[-1]])
                p = 0 if denominator == 0 else len(nd[nd[:,0]==t[0]]) / denominator     #conditional prob
                d.setdefault(t, p)
            self._l.append(d)
        d = {}
        for i in sorted(set(yy)):
            p = len(yy[yy==i]) / len(yy)
            d.setdefault(i, p)
        else: self._d = d; del d

    @property
    def prior_probabilities(self):
        return self._d

    def predict(self, X_pred):
        from numpy import zeros, unique, argmax
        from operator import mul
        from functools import reduce
        l = []
        ct = sorted(unique(self._y.ravel()))
        aProbs = zeros(shape=len(ct), dtype='f')

        for xx in X_pred:
            P_X = reduce(mul, [d[k] for d,k in zip(self._l, xx)])
            for i in ct:
                P_C = self._d.get(i)
                P_XC = reduce(mul, [d.get((k, i), 1) for d,k in zip(self._l, xx)])
                p = (P_C * P_XC) / P_X
                aProbs[i] = p
            l.append(ct[argmax(aProbs)])
        return l
#################################################################################################

def main():
    df = play_golf_table()
    print(df)
    mx,nt = df_into_mx(df)
    X,_,y = np.split(df_into_mx(df)[0], [4,4], axis=1)
    print(X)
    print(y)

    md = NaiveBayesClassifier(X,y)
    print(md.prior_probabilities)

    dfTest = pd.DataFrame([pd.Series(['sunny','cool','high',1,0], index=df.columns),df.iloc[0,:]]).reset_index(drop=True)
    print(dfTest)
    X_test = df_into_mx(dfTest, decoder=nt)[0][:,:-1]
    print(X_test)

    y_pred = md.predict(X_test)
    print(y_pred)

    y_pred = md.predict(X)

    b = np.all(np.equal(y.ravel(), y_pred))
    print(b)
    print(y.ravel().tolist())
    print(y_pred)

    print("========================")

    from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ClassifierMixin
    md = MultinomialNB()
    md = BernoulliNB()
    md = GaussianNB()
    md.fit(X,y)
    y_pred = md.predict(X)
    print(y_pred)

if __name__ == '__main__':main()





