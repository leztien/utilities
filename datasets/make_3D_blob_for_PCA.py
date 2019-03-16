




def make_3D_blob_for_PCA(n_datapoints=500):
    import numpy as np
    def gaussian_kernel(data, landmark='mean', gamma=None):
        from numpy import array, exp
        X = data
        landmark = X.mean(axis=0) if landmark=='mean' else array(landmark)
        assert landmark.shape == (2,), "bad landmark"
        n = len(X)
        a = ((X-landmark)**2).sum(1)
        sqrt_a = a**0.5
        mu = sqrt_a.mean()
        sigma_squared = ((sqrt_a - mu)**2).sum() / (n-1)
        gamma = gamma or 1/(2 * sigma_squared)
        a = exp(-(gamma*a))
        return a

    Σ = [[16,0],
         [ 0,3]]
    M = np.random.multivariate_normal(mean=[0,0], cov=Σ, size=n_datapoints)
    M += np.abs(M.min(axis=0))

    a = gaussian_kernel(M, gamma=0.02)
    a *= np.random.normal(loc=0, scale=a.std(), size=len(M))
    a *= M.std(axis=0).min() * 1.5
    a += np.abs(a.min())
    M = np.hstack([M, a[:,None]])
    nx = np.argsort(M[:,0])
    M = M[nx]
    y = np.arange(1, len(M)+1)

    #rotate the blob
    n = 1/(2**0.5)
    T = [[n,-n,-n],
         [n, n,-n],
         [n, 0, n]]
    M = np.matmul(T,M.T).T
    M += np.abs(M.min(axis=0))
    X = M   # data ready
    return X,y



#####################################################################################

def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    """PCA"""
    X,y = make_3D_blob_for_PCA()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)

    """PLOT"""
    fig = plt.figure(figsize=(12,5))
    sp = fig.add_subplot(121, projection='3d')
    sp.scatter(*X.T, cmap='viridis', c=y)


    sp = fig.add_subplot(122)
    sp.scatter(*X_pca.T, cmap='viridis', c=y)
    sp.axis('equal')
    sp.set_xlabel("PC1");sp.set_ylabel("PC2")

    plt.show()


if __name__=="__main__":main()