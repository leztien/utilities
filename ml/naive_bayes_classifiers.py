
#my naive bayes classifiers


from sklearn.base import BaseEstimator, ClassifierMixin
class  CategoricalNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self._d = None
        self._l = None

    def fit(self, X, y):
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
        return self


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

class GaussianNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        from numpy import ndarray, array, bincount, hstack
        """ERROR CHECKING"""
        if type(X) is not ndarray:
            try:
                X = array(X, dtype='f')
            except:
                raise Exception("unable to turn argument X into an ndarray")
        if type(y) is not ndarray:
            try:
                y = array(y, dtype='i')
            except:
                raise Exception("unable to turn argument y into an ndarray")
        assert all([(len(set(y)) >= 2),  # at least 2 classes needed
                    (X.shape[0] == y.shape[0]),  # number of observations in X must match the length of y
                    (len(X.shape) == 2),  # X must be a 2D array
                    (len(y.shape) == 1),  # y must be a 1D array
                    (1 not in bincount(y)),  # each class must have at least 2 data-points
                    (sorted(set(y)) == [i for i in range(
                        len(set(y)))])]), "Inadequate array(s)"  # the classes must be labels as [0,1,2,...]
        self.X, self.y = X, y

        """FIT: CALCULATE THE MU'S AND SIGMA'S"""
        from pandas import DataFrame
        df = DataFrame(X, columns=[f'x{i}' for i in range(1, X.shape[-1] + 1)])
        GR = df.groupby(by=y)
        self._mus = GR.mean().values[:, None, :]
        self._sigmas = GR.std().values[:, None, :]

        """PROBABILITIES OF CLASSES"""
        self._probabilities_of_classes = bincount(y) / len(y)
        return self

    def predict(self, X):
        if (self.X is None) or (self.y is None): raise Exception("you must fit the model first")
        self._probabilities = self.predict_proba(X)
        y_pred = self._probabilities.argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        if (self.X is None) or (self.y is None): raise Exception("you must fit the model first")
        from numpy import exp, pi, multiply, ndarray
        """ERROR CHECKING"""
        assert all([(type(X) is ndarray), (X.shape[-1] == self.X.shape[-1])]), "inconsistent array"

        nd3D = exp(-(self.X - self._mus) ** 2 / (self._sigmas ** 2 * 2)) * (1 / (self._sigmas * ((pi * 2) ** 0.5)))
        nd2D = multiply.reduce(nd3D, axis=-1).T  # P(X|C)     likelihoods
        mx = nd2D * self._probabilities_of_classes  # the numerator for the Bayes formula (i.e. non-normalized probs)
        ndVerticalArray = mx.sum(axis=-1)[:,None]  # marginal totals = priori probabilities = denominator for the Bayes formula (i.e. normalizer)
        self._probabilities = mx / ndVerticalArray
        return self._probabilities
    """the .score method is inherited from the parent class and is fully functional !!!"""
###############################################################################################################


class MultinomialDistributionNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        from pandas import DataFrame
        from numpy import bincount
        df = DataFrame(X, columns=[f'x{i}' for i in range(1, X.shape[-1] + 1)])
        GR = df.groupby(by=y)
        self._mxProbs = GR.sum().div(GR.sum().sum(1), axis=0).values
        self._class_probs = bincount(y) / len(y)
        return self

    def predict(self, X):
        def _vectorize(X, probabilities, class_probs):
            def _multinomial_distribution_probability(counts_vector, probabilities_vector):  # returns single probability
                from math import factorial
                from functools import reduce
                from operator import mul
                pp = probabilities_vector
                nn = counts_vector
                p = (factorial(sum(nn)) // reduce(mul, (factorial(n) for n in nn))) * reduce(mul, (p ** n for p, n in zip(pp, nn)))
                return p  # returns single probability
            # --------------------end of the innder function--------------------------------------
            from numpy import zeros
            nclasses = self._class_probs.shape[0]
            nd = zeros(shape=(len(X), nclasses))
            for i in range(len(X)):
                for j in range(nclasses):
                    nd[i, j] = _multinomial_distribution_probability(X[i], probabilities[j])
            nx = (nd * class_probs).argmax(axis=-1)
            return nx
        # ----------END OF _vectorize FUNCTION-------------------------------------------------------------------
        y_pred = _vectorize(X, self._mxProbs, self._class_probs)
        return y_pred
#####################################################################################################
#####################################################################################################

"""TESTS"""
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

def make_data_gaussian():
    from io import StringIO
    from pandas import read_table
    s = """\
    1.0 10.5    20.0    0
    1.5 11.5    21.0    0
    2.0 12.0    22.0    0
    5.1 20.0    30.0    1
    5.2 21.5    31.0    1
    5.3 22.5    32.0    1
    7.0 15.0    40.0    2
    7.5 15.7    41.0    2
    8.0 15.8    40.0    2
    10.5 1.5 10.5    3
    11.7 1.6 10.6    3
    12.8 1.7 10.7    3
    9.0  5.0 15.0    3"""

    df = read_table(StringIO(s), delimiter="\s+", header=None,names=['x1','x2','x3','y'])
    X = df.values[:,:-1]
    y = df['y'].values
    return X,y

def make_data_multinomial():
    from io import StringIO
    from pandas import read_table
    s = """\
    10 2 3 0
    9 1 2 0
    2 5 10 1
    1 6 9 1
    2 2 2 2
    3 4 3 2
    1 2 0 2
    20 10 10 3
    21 11 9 3
    """
    df = read_table(StringIO(s), header=None, sep=r'\s+', names=['red', 'green', 'blue', 'class'])
    df.columns.name = 'marbles'
    X = df.values[:, :-1]
    y = df['class'].values
    return X,y

#--------------------------------------------------------------------------------------------

"""MAIN"""
def main():
    import numpy as np, pandas as pd

    """CategoricalNaiveBayesClassifier"""
    df = play_golf_table()
    print(df)
    mx,nt = df_into_mx(df)
    X,_,y = np.split(df_into_mx(df)[0], [4,4], axis=1)
    y = y.ravel()
    print(X)
    print(y)

    md = CategoricalNaiveBayesClassifier().fit(X,y)
    print(md.prior_probabilities)

    dfTest = pd.DataFrame([pd.Series(['sunny','cool','high',1,0], index=df.columns),df.iloc[0,:]]).reset_index(drop=True)
    print(dfTest)
    X_test = df_into_mx(dfTest, decoder=nt)[0][:,:-1]
    print(X_test)

    y_pred = md.predict(X_test)
    print(y_pred)

    y_pred = md.predict(X)
    score = md.score(X,y)
    print(score)


    """GaussianNaiveBayesClassifier"""
    X,y = make_data_gaussian()
    from sklearn.naive_bayes import GaussianNB
    MD = GaussianNB().fit(X,y)
    scoreSK = MD.score(X,y)

    MD = GaussianNaiveBayesClassifier()
    MD.fit(X,y)
    scoreMY = MD.score(X,y)
    print(scoreSK, scoreMY)

    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    X = d.data
    y = d.target

    from sklearn.naive_bayes import GaussianNB
    MD = GaussianNB().fit(X,y)
    scoreSK = MD.score(X, y)

    MD = GaussianNaiveBayesClassifier().fit(X,y)
    scoreMY = MD.score(X,y)
    print(scoreSK, scoreMY)

    y_pred = MD.predict(X)
    probs = MD.predict_proba(X)
    print(y_pred[:10])
    print(probs)


    """MultinomialDistributionNaiveBayesClassifier"""
    X,y = make_data_multinomial()
    print(X,y)

    MD = MultinomialDistributionNaiveBayesClassifier().fit(X, y)
    y_pred = MD.predict(X)
    print(y_pred, MD.score(X,y))  # 100%

    from sklearn.naive_bayes import MultinomialNB
    MD = MultinomialNB().fit(X,y)
    y_pred = MD.predict(X)
    print(y_pred, MD.score(X,y))

if __name__=='__main__':main()





