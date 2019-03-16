from sklearn.base import BaseEstimator, ClassifierMixin


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
        df['y'] = y
        GR = df.groupby(by='y')
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
        ndVerticalArray = mx.sum(axis=-1)[:,
                          None]  # marginal totals = priori probabilities = denominator for the Bayes formula (i.e. normalizer)
        self._probabilities = mx / ndVerticalArray
        return self._probabilities

    """the .score method is inherited from the parent class and is fully functional !!!"""


###############################################################################################################


def make_data():
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
#==================================================================================================

def main():
    X,y = make_data()
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

if __name__=="__main__":main()





