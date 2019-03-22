


__version__ = '1.0'


def linear_regression_with_gradient_descent(X,y, α=0.05, max_iter=1e+5):
    """get the weights for a linear regression model through gradient descent"""
    from numpy import array, ndarray, ones_like, hstack, dot, allclose, matmul
    X,y = (array(a, dtype='f') if not isinstance(a, ndarray) else a for a in (X,y))
    if not (X[:,0]==1).all():
        X = hstack([ones_like(y).reshape(-1,1), X])
    m,n = X.shape; n-=1
    θ = array([0,]*(n+1), dtype='f')

    εε = [0]*(n+1)
    for i in range(int(max_iter)+1):
        for j in range(n+1):
            d = dot(matmul(X,θ)-y, X[:,j]) / m
            θ[j] -= α * d  # d = partial derivative's slope
            εε[j]=d
        if allclose(εε,0, atol=0.001):break
    else:
        from warnings import warn
        warn(f"maximum number of iterations reached: {i}", Warning)
    return θ

#===========================================================================================================

def linear_regression_with_gradient_descent2(X, y):
    """get the weights for a linear regression model through gradient descent.
    this is the fully vectorized version of the preceding code
    i.e. matrix multiplication is implemented (instead of an inner loop, like in the code above)"""
    from numpy import zeros, exp, hstack, ones, allclose, dot, matmul, abs
    m = X.shape[0]
    θ = zeros(shape=X.shape[-1] + 1)
    X = hstack([ones(m)[:, None], X])
    α = 0.05  # learning rate
    k = 5  # how many pockets in the deque aka stack-array should be
    stack = __import__('collections').deque(maxlen=k)

    for loop in range(int(3e+5)):
        theta_old = θ.copy()
        θ -= α * (matmul(matmul(X, θ) - y, X) / m)

        b = allclose(θ, theta_old, rtol=1e-6)
        stack.append(b)

        if len(stack) >= k and set(stack) == {True}:
            break

    this = globals()[__import__('inspect').stack()[0].function]
    this.loop = this.loops = loop  # number of loops
    return (θ)

#==================================================================================================

from sklearn.base import BaseEstimator, RegressorMixin
class LinearRegressionGradientDescent(BaseEstimator, RegressorMixin):
    """Fit linear regression with gradient descent"""
    def __init__(self, scale=True, learning_rate=0.05):
        self.scale = scale
        self.learning_rate = learning_rate
        self._weights = None

    def _scale(self, X):
        from inspect import stack
        caller = stack()[1].function  # name of the calling function as a string
        if caller == 'fit':
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            self._scaler = StandardScaler().fit(X)
        X = self._scaler.transform(X)
        return X

    def fit(self, X, y):
        from numpy import zeros, hstack, ones, allclose, matmul

        X = self._scale(X) if self.scale else X

        m = X.shape[0]
        θ = zeros(shape=X.shape[-1] + 1)
        X = hstack([ones(m)[:, None], X])
        α = self.learning_rate  # learning rate
        k = 5  # how many pockets in the deque aka stack-array should be
        stack = __import__('collections').deque(maxlen=k)

        for loop in range(int(3e+5)):
            theta_old = θ.copy()
            θ -= α * (matmul(matmul(X, θ) - y, X) / m)

            b = allclose(θ, theta_old, rtol=1e-6)
            stack.append(b)

            if len(stack) >= k and set(stack) == {True}:
                break
        else:
            from warnings import warn
            warn("maximum number of loops reached: %g" % loop, Warning)
        self.loops = loop + 1
        self._weights = θ
        return self

    @property
    def weights(self):
        from warnings import warn
        if self._weights is None:
            warn("the model has not yet been fitted", Warning)
        if self.scale == True:
            warn("the weights have been calculated on scaled data", Warning)
        return self._weights

    def predict(self, X):
        from numpy import ones, hstack
        X = self._scale(X) if self.scale else X
        X = hstack([ones(X.shape[0])[:, None], X])
        ypred = X @ self._weights
        return ypred

    def score(self, X, y):
        from numpy import array, float64
        y = array(y, dtype=float64)
        ypred = self.predict(X)

        SSE = ((y - ypred) ** 2).sum()
        SST = ((y - y.mean()) ** 2).sum()
        RSq = 1 - (SSE / SST)
        return RSq

#====================================================================================================

from sklearn.base import BaseEstimator,RegressorMixin
class RidgeRegressionGradientDescent(BaseEstimator,RegressorMixin):
    def __init__(self, alpha=1.0, learning_rate=0.05, scale=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.scale = scale
        self._weights = None

    def _scale(self, X):
        from inspect import stack
        caller = stack()[1].function  # name of the calling function as a string
        if caller=='fit':
            from sklearn.preprocessing import MinMaxScaler,StandardScaler
            self._scaler = StandardScaler().fit(X)
        X = self._scaler.transform(X)
        return X

    def fit(self, X,y):
        from numpy import zeros,hstack,ones,allclose,matmul,dot
        X = self._scale(X) if self.scale else X

        m,n = X.shape
        θ = zeros(shape=X.shape[-1]+1)
        X = hstack([ones(m)[:,None],X])
        α = self.learning_rate  #learning rate
        k = 5      #how many pockets in the deque aka stack-array should be
        stack = __import__('collections').deque(maxlen=k)

        λ = self.alpha   #  λ in sklearn is alpha
        for loop in range(int(3e+5)):
            theta_old = θ.copy()

            θ[0] -= α * dot(matmul(X,θ)-y, X[:,0]) / m
            for j in range(1, n+1):
                θ[j] = θ[j]*(1-α*(λ/m)) - (α * dot(matmul(X,θ)-y, X[:,j]) / m)

            b = allclose(θ, theta_old, rtol=1e-6)
            stack.append(b)

            if len(stack)>=k and set(stack)=={True}:
                break
        else:
            from warnings import warn
            warn("maximum number of loops reached: %g" %loop, Warning)
        self.loops = loop+1
        self._weights = θ
        return self

    @property
    def weights(self):
        from warnings import warn
        if self._weights is None:
            warn("the model has not yet been fitted", Warning)
        if self.scale==True:
            pass
            #warn("the weights have been calculated on scaled data", Warning)
        return self._weights

    def predict(self, X):
        from numpy import ones,hstack
        X = self._scale(X) if self.scale else X
        X = hstack([ones(X.shape[0])[:,None],X])
        ypred = X@self._weights
        return ypred

    def score(self, X,y):
        from numpy import array, float64
        y = array(y, dtype=float64)
        ypred = self.predict(X)

        SSE = ((y-ypred)**2).sum()
        SST = ((y - y.mean())**2).sum()
        RSq = 1 - (SSE/SST)
        return RSq

#=====================================================================================================


def main():
    import numpy as np
    np.set_printoptions(suppress=True, precision=8)

    """MAKE DATA"""
    rs = np.random.RandomState(0)
    X = rs.randint(-10, 10, size=(20, 5)).astype('f')
    X[:, 0] /= 1000
    X[:, 1] /= 100
    X_ = X.copy()
    y = 1000 * X[:, 0] + 100 * X[:, 1] + 10 * X[:, 2] + 1 * X[:, 3] + 0.1 * X[:, 4] + rs.normal(0, 20, size=X.shape[0])

    md = LinearRegressionGradientDescent()
    md.fit(X, y)
    w = md.weights.round(3)
    score = md.score(X, y)
    ypred = md.predict(X)
    print(w, score)

    # compare with sklearn's Linearregression
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    md = LinearRegression().fit(X, y)
    w = [md.intercept_.round(3), *md.coef_.round(3)]
    score = md.score(X, y)
    print(w, score)


    """TEST RIDGE REGRESSION"""
    X = X_
    md = RidgeRegressionGradientDescent(alpha=100, scale=True)
    md.fit(X, y)
    w = md.weights.round(3)
    score = md.score(X, y)
    ypred = md.predict(X)
    print(w, score)

    # compare with sklearn's Linearregression
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    # md = LinearRegression().fit(X,y)
    md = Ridge(alpha=100.0).fit(X, y)
    w = [md.intercept_.round(3), *md.coef_.round(3)]
    score = md.score(X, y)
    print(w, score)


if __name__=='__main__':main()











