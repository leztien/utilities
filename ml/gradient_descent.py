

def gradient_descent(X,y, α=0.05, max_iter=1e+5) -> "vector of weights":
    from numpy import array, ndarray, ones_like, hstack, dot, allclose, matmul
    X,y = (array(a, dtype='f') if not isinstance(a, ndarray) else a for a in (X,y))
    if not (X[:,0]==1).all():
        X = hstack([ones_like(y).reshape(-1,1), X])
    m,n = X.shape; n-=1
    θ = array([0,]*(n+1), dtype='f')

    εε = [0]*(n+1)
    for i in range(int(max_iter)+1):
        for j in range(n+1):
            d = dot(matmul(X,θ)-y, X[:,j]) / len(y)
            θ[j] -= α * d  # d = partial derivative's slope
            εε[j]=d
        if allclose(εε,0, atol=0.000001):break
    else:
        from warnings import warn
        warn(f"maximum number of iterations reached: {i}", Warning)
    return θ

#============================================================================

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

def qq(a):
    from scipy.stats import norm
    from statistics import mean, stdev
    μ ,σ = mean(a), stdev(a)
    a = [( v -μ ) /σ for v in sorted(a)]
    n = len(a)
    p = 1/ (n + 1)
    pp = [p * n for n in range(1, n + 1)]
    zz = norm.ppf(pp, loc=0, scale=1)
    return (a, zz)
#===================================================================

def main():
    from numpy import array
    from numpy.random import multivariate_normal
    from matplotlib.pyplot import show, subplots
    from seaborn import pairplot
    from pandas import DataFrame
    from myutils.datasets import make_legitimate_covariance_matrix

    #make data
    cm = [[7, -5, 4,  8],      # RSq will be about 0.85
          [-5,14,-9,  0],
          [4, -9, 15,-5],
          [8,  0,-5, 19]]
    cm = array(cm, dtype='i')
    cm = cm | cm.transpose()

    cm  = make_legitimate_covariance_matrix(ndim=4)    # use this covariance matrix instead of the above one

    mx = multivariate_normal(mean=[0,0,0,0], cov=cm, size=100)




    mx = scale(mx)
    X,y = split(mx)


    #model
    from sklearn.linear_model import LinearRegression
    md = LinearRegression(fit_intercept=True, normalize=False).fit(X,y)
    score = md.score(X,y)
    print("RSq =",score)
    w = md.coef_
    w0 = md.intercept_
    print("intercept, weights = ", w0,w)

    #vizualize data
    df = DataFrame(mx, columns=[f'feature {i+1}' for i in range(4)])
    pairplot(df)

    fig, SP = subplots(1,2, figsize=(10,5))
    sp = SP[0]
    y_true, y_pred = array(y), md.predict(X)
    sp.plot(y_true, y_pred, 'k.')
    sp.axis('equal')
    sp.set(xlabel="y-true", ylabel="y-predicted", title="true vs predicted values")

    sp = SP[-1]
    εε = y_true - y_pred
    εε,zz = qq(εε)
    sp.plot(εε,zz, 'k.')
    sp.axis('equal')
    sp.set(xlabel="errors normalized", ylabel="qq-values", title="qqplot of errors")

    x0,*w = gradient_descent(X,y)
    print(x0, w)

    show()

if __name__=='__main__':main()


