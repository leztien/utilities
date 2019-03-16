

class MyPCA(object):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self._means = None

    def fit(self, X):
        from  numpy import array, cov, argsort
        from scipy.linalg import eig
        assert 'ndarray' in str(type(X)) and len(X.shape)==2 and set(X.shape).intersection([0,1])==set(), "the argument must be an ndarray of shape m by n"
        self.X = X.copy()
        self._means = self.X.mean(axis=0)
        self.X = self.X - self._means
        E = cov(self.X.T)
        eigenvalues, eigenvectors = eig(E)
        eigenvectors = eigenvectors.T
        eigenvalues = array([e.real for e in eigenvalues], dtype='f')
        nx = argsort(eigenvalues)[::-1][:self.n_components]
        eigenvalues, eigenvectors = (a[nx] for a in (eigenvalues, eigenvectors))
        #eigenvectors[eigenvectors.diagonal()<0] = -eigenvectors[eigenvectors.diagonal()<0]
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        return self

    def transform(self, X):
        if self._means is None:
            raise Exception("you must fit the model first")
        self.X = X.copy()
        self.X = self.X - self._means
        X_transformed = (self.X[None,:,:] * self.eigenvectors[:,None,:]).sum(axis=-1).T
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed
#===========================================================================================






def main():

    from numpy.random import multivariate_normal
    import matplotlib.pyplot as plt
    sp = plt.axes()

    cm = [[1,2],
          [2,5]]
    X = multivariate_normal(mean=[10,20], cov=cm, size=10)
    n_components = 2
    md = MyPCA(n_components=n_components)
    md.fit(X)
    X_transformed = md.transform(X)
    sp.scatter(*X_transformed.T, color='green', alpha=0.5)

    """TEST"""
    from sklearn.decomposition import PCA
    md = PCA(n_components=n_components)
    X_transformed = md.fit_transform(X)
    sp.scatter(*X_transformed.T, color='red', alpha=0.5, s=50)


    """TEST2"""
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    X = d.data
    y = d.target

    md = MyPCA(n_components=1)
    md.fit(X)
    X_tr = md.transform(X)
    print(X_tr.shape)

    from sklearn.linear_model import LogisticRegression
    md = LogisticRegression()
    md = md.fit(X_tr, y)
    y_pred = md.predict(X_tr)

    from sklearn.metrics import accuracy_score
    y_true = y
    p = accuracy_score(y_true, y_pred)
    print(p)

    """real PCA"""
    md = PCA(n_components=1)
    md.fit(X)
    X_tr2 = md.transform(X)
    md = LogisticRegression()
    md = md.fit(X_tr2, y)
    y_pred = md.predict(X_tr2)
    y_true = y
    p = accuracy_score(y_true, y_pred)
    print(p)

    plt.show()
if __name__=='__main__':main()






















