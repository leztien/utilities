
#UNFINISHED

class RegularizedRegression():
    def __init__(self, model='Ridge', alpha=1.0):
        self.X = None
        self.y = None
        self.intercept = None
        self.weights = None
        self.mse = None
        self._mse_history = None
        self.model = model.lower()
        self.alpha = alpha
        assert self.model in ('ols','ridge','lasso'), f"{model} is nor supported."


    class DialGranular(float):
        def __init__(self, *args):
            self.increment = 0.1

        def __str__(self):
            return self.__repr__() + "°"

        def _validate(self, v):
            if v % 180 == 0:
                v = 0.
            elif v > 180:
                v = v % 180
            elif v < 0:
                v = v % 180
            assert v >= 0
            return v

        def _add(self, arg):
            v = float(self) + float(round(arg, 0)) * self.increment
            v = self._validate(v)
            if self.skip90 and v == 90.: v += self.increment
            return self.__class__(v)

        def _mul(self, arg):
            v = int(self) * int(round(arg, 0))
            v = self._validate(v)
            if self.skip90 and v == 90: v = 91
            return self.__class__(v)

        def _sub(self, arg):
            v = float(self) - float(round(arg, 0)) * self.increment
            v = self._validate(v)
            if self.skip90 and v == 90: v -= self.increment
            return self.__class__(v)

        def _div(self, arg):
            v = int(self) // int(round(arg, 0))
            v = self._validate(v)
            if self.skip90 and v == 90: v = 89
            return self.__class__(v)

        def __add__(self, arg):
            return self._add(arg)

        def __iadd__(self, arg):
            return self._add(arg)

        def __radd__(self, arg):
            return self._add(arg)

        def __mul__(self, arg):
            return self._mul(arg)

        def __imul__(self, arg):
            return self._mul(arg)

        def __rmul__(self, arg):
            return self._mul(arg)

        def __sub__(self, arg):
            return self._sub(arg)

        def __isub__(self, arg):
            return self._sub(arg)

        def __rsub__(self, arg):
            raise Exception("reverse subtraction not defined")

        def __floordiv__(self, arg):
            return self._div(arg)

        def __truediv__(self, arg):
            return self._div(arg)

        def __idiv__(self, arg):
            return self._div(arg)

        def __itruediv__(self, arg):
            return self._div(arg)

        def __ifloordiv__(self, arg):
            return self._div(arg)

        def __rdiv__(self, arg):
            raise Exception("reverse division not defined")

        def __rfloordiv__(self, arg):
            raise Exception("reverse division not defined")

        def __rtruediv__(self, arg):
            raise Exception("reverse division not defined")

        def __abs__(self):
            self.skip90
            return float(self)

        @property
        def int_value(self):
            return self.__int__()

        @property
        def skip90(self):
            if '_skip90' not in self.__dict__:
                self._skip90 = True
            return self._skip90

        @skip90.setter
        def skip90(self, arg):
            assert isinstance(arg, bool), "skip90 must be boolean"
            self._skip90 = arg

        @skip90.deleter
        def skip90(self):
            self._skip90 = False

    # ----END OF THE DIAL_GRANULAR CLASS----

    def _weights_from_slope_angles(self, slope_angles):
        from numpy import deg2rad, tan
        assert hasattr(slope_angles, '__iter__'), "you must provide an array of slope angles"
        assert len(slope_angles) >= 2
        weights = tan(deg2rad(slope_angles))
        intercept = self._ymean - (self._Xmeans * weights).sum()
        return (intercept, weights)   #returns intercept and weights

    def _SSE(self, slope_angles, return_intercept_and_weights=False):
        from numpy import concatenate, abs
        assert hasattr(slope_angles, '__iter__'), "you must provide an array of slope angles"
        assert len(slope_angles) == self.X.shape[-1], "you must provide {} angles, you provided {}".format(self.X.shape[-1], len(slope_angles))
        intercept, weights = self._weights_from_slope_angles(slope_angles)
        yy_true = self.y
        yy_pred = (self.X*weights).sum(axis=1) + intercept
        ##penalty = (weights**2).sum() * self.alpha
        weights = concatenate([[1,], weights])
        penalty = 0 if self.model == 'ols' else (weights**2).sum()*self.alpha if self.model == 'ridge' else abs(weights).sum()*self.alpha if self.model == 'lasso' else None
        sse = ((yy_true - yy_pred) ** 2).sum() + penalty
        return sse if not return_intercept_and_weights else (sse, intercept, weights)


    def fit(self, X,y):
        assert all('ndarray' in str(type(e)) for e in (X,y)), "the arguments must be ndarrays"
        assert all(hasattr(e, '__iter__') for e in (X,y)), "both arguments must be arrays"
        assert len(X)==len(y), "both arrays must be of equal length"
        assert X.shape[0] == len(y), "non allignable arrays"
        from numpy import zeros, argsort
        self.X = X
        self.y = y
        self._ymean = self.y.mean()
        self._Xmeans = self.X.mean(axis=0)

        l = []  #history of mse's
        table = zeros(shape=(3,self.X.shape[1]+1), dtype='f')   #table columns:  mse  slope_angle_1  slope_angle_2  slope_angle_3 ...
        table[0,0] = self._SSE(table[0, 1:])

        k = 0   #safety break
        n = self.X.shape[1] * 3     #how many mse's at the tail of the history list must be the same
        while True:
            for dim in range(1, self.X.shape[1]+1):
                table[1:] = table[0]
                dial = self.DialGranular(table[0,dim])
                table[1,dim] = abs(dial + 1)
                table[2, dim] = abs(dial - 1)
                table[1,0] = self._SSE(table[1, 1:])
                table[2,0] = self._SSE(table[2, 1:])
                table = table[argsort(table[:,0])]
                l.append(table[0,0])
            if len(set(l[-n:]))==1: break
            k += 1
            if k > 360 * self.X.shape[1]:
                print("SAFETY BREAK")
                break
            else: pass#print(f"finished loop no. {k}", table)

        self.intercept, self.weights = self._weights_from_slope_angles(table[0,1:])
        self.mse = table[0,0]
        self._mse_history = l
        return self

    def mse_history_plot(self, ax=None):
        from matplotlib.pyplot import gca
        sp = ax or gca()
        l = self._mse_history
        sp.plot(l)
        return sp
#============END OF THE FitHyperplaneByGradientDescent CLASS=================================================================


class Make3DDataset():
    def __init__(self, means=None, ndim=3, size=None):
        from sklearn.linear_model import LinearRegression
        from numpy import cov, meshgrid, zeros
        from numpy.random import multivariate_normal, randint
        from matplotlib.pyplot import figure
        from mpl_toolkits.mplot3d import Axes3D

        assert isinstance(ndim, int), "number of dimensions must be an integer"
        means = means or [10]*ndim
        assert len(means) == ndim, "length of means must be equal to the number of dims, you got: length of means {}, number of dimensions {}".format(len(means), ndim)

        def _make_cov_mx():
            cm = zeros(shape=(ndim,ndim), dtype='int8')
            while True:
                random_ints = sorted(randint(low=1, high=ndim*2+6, size=ndim))
                if len(set(random_ints)) >= ndim: break
            [cm.__setitem__((i,slice(i+1,None)), random_ints[i]) for i in range(ndim)]
            cm = cm + cm.T
            [cm.__setitem__((i,i), cm[i,i+1]) for i in range(ndim-1)]
            cm[-1,-1] = random_ints[-1]
            return cm

        self.DATASET = multivariate_normal(mean=means, cov=_make_cov_mx(), size=int(size or 100))
        self._means = self.DATASET.mean(axis=0)
        self._covariance_matrix = cov(self.DATASET.T)
        self.X = self.DATASET[:,:-1]
        self.y = self.DATASET[:,-1]

        md = LinearRegression().fit(self.X, self.y[:,None])
        self.weights = md.coef_[0]
        self.intercept = md.intercept_[0]

        if ndim==3:
            w0,(w1,w2) = self.intercept, self.weights
            X,Y = meshgrid(sorted(self.X[:,0])[::len(self.X[:,0])-1], sorted(self.X[:,1])[::len(self.X[:,1])-1])
            Z = X*w1 + Y*w2 + w0
            self.sp = Axes3D(figure())
            self.sp.scatter(*self.DATASET.T, marker='.', color='k')
            self.xbar, self.ybar, self.zbar = self.DATASET.mean(axis=0)
            self.sp.scatter(self.xbar, self.ybar, self.zbar, marker='o', color='red', s=60)
            self.sp.plot_surface(X,Y,Z, color='gray', alpha=0.2)
            self.sp.plot_wireframe(X, Y, Z, color='blue', alpha=0.6)
            self.sp.set_xlabel('x-axis');self.sp.set_ylabel('y-axis');self.sp.set_zlabel('z-axis')

    @property
    def covariance_matrix(self):
        return self._covariance_matrix
    @property
    def means(self):
        return self._means

    def add_plane(self, *args, **kwargs):
        """A,B,C,D  must be provided from this form Ax+By+Cz+D=0 (mind the sign of D)  OR use this form: w0,w1,w2"""
        assert 4 >= len(args) >= 3, "either 4 or 3 arguments are needed. you provided {}".format(len(args))
        assert self.X.shape[-1]==2, "can add a plane to a 3D dataset only"
        from numpy import meshgrid
        X, Y = meshgrid(sorted(self.X[:, 0])[::len(self.X[:, 0]) - 1], sorted(self.X[:, 1])[::len(self.X[:, 1]) - 1])
        if len(args)==3:
            w0, w1, w2 = args
            Z = X * w1 + Y * w2 + w0
        elif len(args)==4:
            a,b,c,d = args
            Z = (X*-a + Y*-b - d) / c           # the sign of d is still in question
        else: raise Exception("unforseen error")
        color = kwargs.get('color', 'lightblue')
        self.sp.plot_surface(X, Y, Z, color=color, alpha=0.6)
        return self.sp

    def show(self):
        if self.X.shape[-1] != 2: exit()
        from matplotlib.pyplot import show
        show()
############### END OF Make3DDataset ##################################################################################


def main():
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=250)

    import matplotlib.pyplot as plt
    ds = Make3DDataset(ndim=3)
    cm = ds.covariance_matrix
    X = ds.X
    y = ds.y
    print("true intercept, weights:", ds.intercept, ds.weights)

    md = RegularizedRegression()
    md = md.fit(X,y)
    print("gradient descent intercept and weights:", md.intercept, md.weights)
    ds.add_plane(md.intercept, *md.weights, color='magenta')
    #md.mse_history_plot(ax=plt.axes())




    ds.show()



    print()
    """TEST RIDGE"""
    alpha = 10000

    from sklearn.linear_model import Ridge, Lasso
    md = Ridge(alpha=alpha)
    md = md.fit(X,y)
    print("sk Ridge intercept and weights:", md.intercept_, md.coef_)

    md = RegularizedRegression(model='Ridge', alpha=alpha)
    md = md.fit(X,y)
    print("my Ridge intercept and weights:", md.intercept, md.weights)

    #Ridge test#2
    alpha = 100
    ds = Make3DDataset(ndim=10)
    X = ds.X
    y = ds.y

    md = Ridge(alpha=alpha)
    md = md.fit(X,y)
    print("\nsk Ridge intercept and weights:", md.intercept_, md.coef_)

    md = RegularizedRegression(model='Ridge', alpha=alpha)
    md = md.fit(X,y)
    print("my Ridge intercept and weights:", md.intercept, md.weights)


    """TEST LASSO"""
    alpha = 1
    ds = Make3DDataset(ndim=10)
    X = ds.X
    y = ds.y

    md = Lasso(alpha=alpha)
    md = md.fit(X,y)
    print("\nsk Lasso intercept and weights:", md.intercept_, md.coef_)

    md = RegularizedRegression(model='Lasso', alpha=alpha)
    md = md.fit(X,y)
    print("my Lasso intercept and weights:", md.intercept, md.weights)




if __name__=='__main__':main()





