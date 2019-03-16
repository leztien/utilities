from sklearn.base import BaseEstimator, ClassifierMixin


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


def make_data():
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


def main():
    X,y = make_data()
    print(X,y)

    MD = MultinomialDistributionNaiveBayesClassifier().fit(X, y)
    y_pred = MD.predict(X)
    print(y_pred, MD.score(X,y))  # 100%

    from sklearn.naive_bayes import MultinomialNB
    MD = MultinomialNB().fit(X,y)
    y_pred = MD.predict(X)
    print(y_pred, MD.score(X,y))

if __name__=='__main__':main()












