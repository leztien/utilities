







class PseudoPCA():
    def __init__(self):
        self._slope = None
        self._intercept = None

    @staticmethod
    def orthogonal_distance(x, y, slope, intercept):
        from math import sin, radians, degrees, atan
        slope_in_degrees = abs(degrees(atan(slope)))
        orth_dist = abs(y - (x * slope + intercept)) * sin(radians(90 - slope_in_degrees))
        return orth_dist

    def fit_transform(self, X):
        if 'ndarray' not in str(type(X)):
            from numpy import array
            X = array(X, dtype='f')
        assert 'ndarray' in str(type(X)), "must be a numpy ndarray"
        assert len(X.shape)==2, "must be a 2D ndarray"
        assert all(X.shape[i]>1 for i in (0,1)), "must be a matrix"
        from scipy.stats import linregress
        from functools import partial
        from numpy import vectorize, sqrt
        self.X = X
        xx,yy = self.X.T
        self._slope, self._intercept, *_ = linregress(xx,yy)

        fp = partial(self.orthogonal_distance, slope=self._slope, intercept=self._intercept)
        fv = vectorize(fp)
        orthogobal_distances = fv(xx,yy) #orthogonal distances
        xbar,ybar = X.mean(axis=0)
        hypotenuses_squared = (xx - xbar)**2 + (yy - ybar)**2
        a = sqrt( hypotenuses_squared - orthogobal_distances**2 )
        a[xx < xbar] = -a[xx < xbar]    #this calculation is not 100% correct
        return a

    @property
    def slope(self):
        from logging import info, INFO, basicConfig
        basicConfig(level=INFO, format="%(message)s")
        if self._slope is None:
            info("you must fit the model first")
        return self._slope

    @property
    def intercept(self):
        from logging import info, INFO, basicConfig
        basicConfig(level=INFO, format="%(message)s")
        if self._intercept is None:
            info("you must fit the model first")
        return self._intercept

def main():
    from numpy import argsort, set_printoptions
    set_printoptions(suppress=True)
    from numpy.random import seed, multivariate_normal
    from scipy.stats import linregress
    import matplotlib.pyplot as plt

    mxCov = [[2, 3],
             [3, 5]]
    X = multivariate_normal(mean=[10, 15], cov=mxCov, size=25)
    X = X[argsort(X[:,0])]
    print(X)

    sp = plt.axes()
    sp.scatter(*X.T, marker='.', s=10, color='k')
    sp.scatter(*X.mean(axis=0), marker='o')
    slope,intercept,*_=linregress(*X.T)
    f1 = lambda x : x*slope+intercept
    x1,x2 = min(X[:,0]), max(X[:,0])
    sp.plot([x1,x2],[f1(x1),f1(x2)], linewidth=.5, color='gray')
    #sp.axis([6,max(X.ravel())+1, 6,max(X.ravel())+1])
    [sp.text(x,y, i+1, fontdict={'fontsize':7}) for i,(x,y) in enumerate(zip(*X.T))]

    md = PseudoPCA()
    a = md.fit_transform(X)
    print("slope, intercept:", md.slope, md.intercept)
    print(a)

    #test compare
    from sklearn.decomposition  import PCA
    md = PCA(n_components=1)
    nd = md.fit_transform(X)
    print(nd.reshape(1,-1)[0])

    plt.show()
if __name__=='__main__':main()