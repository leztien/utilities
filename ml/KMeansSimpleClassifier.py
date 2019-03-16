

import numpy as np, matplotlib.pyplot as plt
np.random.seed(2)

class KMeansClassifier():
    def __init__(self, n_clusters=2, random_state=None):
        self.n_clusters = n_clusters
        self._classes = []
        self._centroids = []
        if random_state is not None: np.random.seed(random_state)

    def fit(self, X):
        points = X[np.random.choice(len(X), size=self.n_clusters, replace=False)]
        points_history = np.zeros_like(points)

        """LOOP"""
        for i in range(100):
            slope = np.divide.reduce(np.subtract.reduce(np.fliplr(points), axis=0)[None,:], axis=1)[0]
            x,y = points[0]

            """orthogonal line"""
            slope_orthogonal = -1/slope
            xbar,ybar = points.mean(axis=0)
            intercept_orthogonal = ybar - (xbar*slope_orthogonal)

            """determine the line function"""
            func = np.vectorize(lambda x,y : y >= (x*slope_orthogonal + intercept_orthogonal))

            """assign the points"""
            y = np.where(func(*X.T), 0,1)

            """calculate the centroids"""
            from pandas import DataFrame
            points = DataFrame(X, dtype='f', columns=['x','y']).assign(c=y).groupby('c').mean().values
            if np.allclose(points.ravel(), points_history.ravel()): break
            else: points_history = points
        else: "ran out of iterations"
        self._classes = y
        self._centroids = points
        return self

    def predict(self, X):
        print(self.centroids_)
        C = self.centroids_
        y = np.argmax(((X[None,:,:] - C[:,None,:])**2).sum(axis=-1).T, axis=1)
        return y


    @property
    def classes_(self):
        return self._classes
    @property
    def centroids_(self):
        return self._centroids
#==========================================================================================


def make_dataset(size=20, random_state=None):
    from random import uniform, choice
    if random_state is not None: np.random.seed(random_state)
    xx = [uniform(3,9) for _ in range(2)]
    yy = 3,9
    xx,yy = (xx,yy) if choice([True,False]) else (yy,xx)
    blob_1_means, blob_2_means = zip(xx,yy)
    t = ((1,0,1,0),(1,2,2,5),(1,1,1,2),(1,2,2,5),(1,-2,-2,5))
    cm1 = np.array(choice(t), dtype=np.int8).reshape(2,2)
    cm2 = np.array(choice(t), dtype=np.int8).reshape(2, 2)
    blob_1 = np.random.multivariate_normal(mean=blob_1_means, cov=cm1, size=size)
    blob_2 = np.random.multivariate_normal(mean=blob_2_means, cov=cm2, size=size)
    M = np.vstack([blob_1,blob_2])
    M = M[np.argsort(M[:,0])]
    return M
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def main():
    M = make_dataset(random_state=42)

    md = KMeansClassifier(n_clusters=2, random_state=42)
    md = md.fit(M)
    y = md.classes_
    nd = md.centroids_

    plt.scatter(*M.T, c=y, cmap=plt.cm.RdBu)
    plt.scatter(*nd.T, color='red', s=50)

    plt.show()


    y_pred = md.predict(M)

if __name__=='__main__':main()






























