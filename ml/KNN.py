



class KnnClassifier():
    def __init__(self, n_neighbors=1):
        self._n = n_neighbors

    def fit(self, X,y):
        self._X = X
        self._y = y
        return self

    def _func(self, mx):
        from numpy import bincount, zeros
        y = zeros(shape=len(mx), dtype='i')
        for i,eRow in enumerate(mx): y[i] = bincount(eRow).argmax()
        return y

    def predict(self, X):
        from numpy import argpartition
        self._X_pred = X
        nd = ((self._X[None,:,:] - self._X_pred[:,None,:])**2).sum(axis=2)
        nx = argpartition(nd, axis=1, kth=self._n-1)[:,:self._n]
        nd = self._y.take(nx)
        y = self._func(nd)
        return y


class KnnRegressor():
    def __init__(self, n_neighbors=1):
        self._n = n_neighbors

    def fit(self, X, y):
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        self._X_pred = X
        assert self._X.shape[-1] ==self._X_pred.shape[-1], "array dims must match"
        from numpy import argpartition, zeros

        mxDistances = ((self._X[None,:,:] - self._X_pred[:,None,:])**2).sum(axis=2)
        NX = argpartition(mxDistances, axis=1, kth=self._n-1)[:, :self._n]
        y_pred = zeros(shape=len(self._X_pred), dtype='f')

        for i,nx in enumerate(NX):
            y = self._y[nx].mean()
            y_pred[i] = y
        return y_pred
#=====================================



def main():     #TEST
    from mglearn.datasets import make_forge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_forge()
    X_train,X_test, y_train, y_test = train_test_split(X,y, random_state=0)

    n = 5
    md = KnnClassifier(n_neighbors=n)
    md.fit(X_train,y_train)
    y_pred_my = md.predict(X_test)

    md = KNeighborsClassifier(n_neighbors=n)
    md.fit(X_train,y_train)
    y_pred_sk = md.predict(X_test)

    ab = y_pred_my == y_pred_sk
    print(ab.all())

    ########################
    import numpy as np
    X = np.random.randint(0,100, size=(50,3))
    y = np.random.randint(0,3, size=len(X))

    X_train,X_test, y_train, y_test = train_test_split(X,y, random_state=0)

    n = 7
    md = KnnClassifier(n_neighbors=n)
    md.fit(X_train,y_train)
    y_pred_my = md.predict(X_test)

    md = KNeighborsClassifier(n_neighbors=n)
    md.fit(X_train,y_train)
    y_pred_sk = md.predict(X_test)

    ab = y_pred_my == y_pred_sk
    print(ab.all())

    ############################################

    from pylab import scatter, plot, show
    from sklearn.neighbors import KNeighborsRegressor
    from mglearn.datasets import make_wave
    X,y = make_wave()
    import numpy as np
    X_test = np.arange(-3,3,0.001).reshape(-1,1)

    n=4
    md = KNeighborsRegressor(n_neighbors=n).fit(X,y)
    scatter(X.ravel(), y, marker='.', color='blue')

    y_pred = md.predict(X_test)
    plot(X_test.ravel(), y_pred, linewidth=0.4)

    md = KnnRegressor(n_neighbors=n).fit(X,y)
    y_pred = md.predict(X_test)
    plot(X_test.ravel(), y_pred, linewidth=0.6, linestyle='--', alpha=0.5, color='red')

    show()

    #------------------------------------

    X,y = np.split(np.random.randint(-10,10, size=(10,5)), axis=1, indices_or_sections=[4])
    y = y.ravel()

    n = 5
    md = KnnRegressor(n_neighbors=n).fit(X,y)
    X_test = np.random.randint(-10,10,size=(25,4))
    y_pred_my = md.predict(X_test)

    md = KNeighborsRegressor(n_neighbors=n).fit(X,y)
    y_pred_sk = md.predict(X_test)

    b = np.allclose(y_pred_my, y_pred_sk)
    print(b)


if __name__=="__main__":main()








