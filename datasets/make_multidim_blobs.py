
def make_multidim_blobs(n_blobs=3, n_points=100, n_dim=3, range=100, relative_dispersion = 10):
    import builtins
    from numpy import zeros, float16, warnings, diag, abs, uint8, argsort
    from numpy.random import randint, multivariate_normal

    m = n_points // n_blobs
    working_range = 100

    σ2 = (working_range / (n_blobs + 1) ** (1 / n_dim) / relative_dispersion) ** 2
    σ2 = int(σ2*0.5), int(σ2*1.5)

    X = zeros(shape=(m * n_blobs, n_dim), dtype=float16)
    y = zeros(shape=X.shape[0], dtype=uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for i in builtins.range(n_blobs):
            while True:
                diagonal = randint(*σ2, size=n_dim)
                mx = diag(diagonal)
                [mx.__setitem__([i, slice(0,i,None)], randint(-1,1, size=n_dim)[:i]) for i in builtins.range(n_dim)]
                Σ = mx | mx.T
                μ = randint(0, working_range, size=n_dim)
                try:
                    mx = multivariate_normal(mean=μ, cov=Σ, size=m)
                    break
                except RuntimeWarning:
                    continue
            X[i*m:i*m+m,:] = mx
            y[i * m:i * m + m] = i
    #the last touches
    X += abs(X.min())
    X *= range/X.max()
    nx = argsort(X[:,0])
    X = X[nx]
    y = y[nx]
    return X,y


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    X,y = make_multidim_blobs(n_blobs=5, n_points=250, n_dim=3, range=1, relative_dispersion=10)

    sp = plt.subplot(111, projection='3d')
    sp.scatter(*X.T, c=y)

    print(X)
    plt.show()

if __name__=='__main__':main()










