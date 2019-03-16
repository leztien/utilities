



class GradientDescent:
    def __init__(self, α=0.01, **kwargs):
        self.α = α
        self.max_iter = kwargs.get('max_iter',None) or int(1e+5)+1
        self._weights = None
        self._J_values = []
        self._h = lambda X : self.ModelNotFitted()
        self._J_slope = []

    def fit(self, X,y):
        from numpy import array, ndarray, ones_like, ones, zeros, hstack, dot, allclose, matmul
        X,y = (array(a, dtype='f') if not isinstance(a, ndarray) else a for a in (X,y))
        if not (X[:,0]==1).all():
            X = hstack([ones_like(y).reshape(-1,1), X])
        m,n = X.shape; n-=1

        θ = ones(shape=n+1, dtype='f')
        εε = zeros(shape=n+1, dtype='f')

        #cost function J
        J = lambda θ : ((matmul(X,θ)-y)**2).sum() / (2*m)

        """ITERATOR"""
        for i in range(self.max_iter):
            for j in range(n+1):
                d = dot(matmul(X,θ)-y, X[:,j]) / len(y)
                θ[j] -= self.α * d  # d = partial derivative's slope
                εε[j]=d
            else:  # after each inner iteration
                self._J_values.append( J(θ) ) #log the cost function value
                if self._has_converged(i): break  #check for convergense TODO: correct
        else: #after reaching the max_iter
            from warnings import warn
            warn(f"maximum number of iterations reached: {i}", Warning)
        self._weights = θ
        self._h = lambda X_new : matmul(X_new, self._weights)
        return self

    def _has_converged(self, iteration_number):
        from numpy import array
        n = 10  # n last values
        if iteration_number < n: return False
        yy = array(self._J_values[-n:])
        slope = (yy[1:] - yy[:-1]).sum() / (n-1)
        slope = (slope + ((yy[-1]-yy[0])/(n-1))) / 2   # combining two slopes
        b = (-1e-12 < slope <= 0)
        self._J_slope.append(slope)
        #TODO: changing alpha
        return b

    def predict(self, X):
        from numpy import array, ndarray,hstack, ones
        if not isinstance(X, ndarray):
            X = array(X)
        if X.shape[1] == len(self._weights)-1:
            X = hstack([ones(shape=X.shape[0]).reshape(-1,1), X])
        elif X.shape[1] == len(self._weights):
            pass
        else: raise TypeError("dims do not match")
        return self._h(X)

    def score(self,X,y):  #coefficient of determination = RSq
        from numpy import array
        y_true, y_pred = array(y), self.predict(X)
        ybar = y_true.mean()
        SSR = ((y_pred - ybar)**2).sum()
        SST = ((y_true - ybar)**2).sum()
        RSq = SSR/SST
        return RSq


    @property
    def cost_vs_iterations(self):
        return self._J_values

    class ModelNotFitted(Exception):pass
    @property
    def intercept_(self):
        if self._weights is None:
            raise self.ModelNotFitted("you must fit the model first")
        else:
            return self._weights[0]

    @property
    def coef_(self):
        if self._weights is None:
            raise self.ModelNotFitted("you must fit the model first")
        else:
            return self._weights[1:]



#===================================================================================

def transpose(mx):
    assert len(mx)>1 and len(set(len(a) for a in mx))==1, "bad args"
    m,n = len(mx), len(mx[0])
    new = [[0,]*m for _ in range(n)]
    [new[j].__setitem__(i, mx[i][j]) for i in range(m) for j in range(n)]
    return new

def vectorize(func):
    def _func(mx):
        from numpy import ndarray, matrix
        if type(mx) in (ndarray, matrix):
            mx = mx.tolist()
        assert isinstance(mx,list), "must be a list"
        mx = transpose(mx)
        for a in mx:
            a[:] = func(a)
        mx = transpose(mx)
        return mx
    return _func

@vectorize
def scale(a):
    mn,mx = min(a),max(a)
    r = mx-mn
    a = [(x-mn)/r for x in a]
    return a

def split(mx):
    global transpose
    mx = transpose(mx)
    X,y = transpose(mx[:-1]), mx[-1]
    return X,y

def make_data(m=100):  # scaled data
    from numpy import array
    from numpy.random import multivariate_normal
    from myutils.datasets import make_legitimate_covariance_matrix
    Σ = [[7, -5, 4,  8],      # RSq will be about 0.85
         [-5,14,-9,  0],
         [4, -9, 15,-5],
         [8,  0,-5, 19]]
    Σ = array(Σ, dtype='i')
    Σ = Σ | Σ.transpose()

    Σ  = make_legitimate_covariance_matrix(ndim=4)    # use this covariance matrix instead of the above one

    mx = multivariate_normal(mean=[0,0,0,0], cov=Σ, size=m)

    mx = scale(mx)
    X,y = split(mx)
    return X,y
#===========================================================================

def main():
    X,y =make_data()
    md = GradientDescent()
    md.fit(X,y)
    print("my score =", md.score(X,y))
    print("my weights:", md.intercept_, md.coef_)


    """COMPARE WITH SKLEARN"""
    from sklearn.linear_model import LinearRegression
    MD = LinearRegression()
    MD.fit(X,y)
    print("sklearn score =", MD.score(X,y))
    print("sklearn weights:", MD.intercept_, MD.coef_)

    """VIZUALIZE COST REDUCTION VS ITERATIONS"""
    import matplotlib.pyplot as plt
    jj = md.cost_vs_iterations
    n = len(jj)

    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.plot(range(1,n+1), jj)
    sp.set(title="cost function vs iterations", xlabel="iterations", ylabel="J")
    plt.show()

    #another test
    x = tuple(range(1, 11))
    x2 = tuple(x ** 2 for x in x)  # x squared
    y = tuple(x2 * 2 + 10 for x2 in x2)

    X = transpose([x, x2])
    X = scale(X)

    md = GradientDescent().fit(X, y)
    intercept, (weight_x, weight_x2) = md.intercept_, md.coef_

    print(intercept, weight_x, weight_x2)
    print(md.score(X, y))

    y_true = y
    y_pred = md.predict(scale(X))
    print(y_true)
    print(y_pred.round(1))


if __name__=='__main__':main()

























