#!/usr/bin/python3

"""
creates 9 almost uniform 3D blobs
for cluster analysis or supervised ML
"""

import builtins

__version__ = "1.0.1"

def make_9_3D_blobs(n_samples=1000):
    """docs"""
    import numpy as np
    from itertools import permutations
    Σ = [[1,0,0], [0,1,0], [0,0,1]]
    X = np.random.multivariate_normal(mean=[0,0,0], cov=Σ, size=n_samples)
    g = permutations([-4,4]*3, 3)
    μμ = np.array(list(set(g)), dtype=np.int8)
    p = 1/9
    for i,μ in enumerate(μμ):
        X[int(len(X)*p)*i:int(len(X)*p)*(i+1)] += (μ + np.random.randn(3))
    X += np.abs(X.min())
    y = sum(([n]*int(len(X)*p) for n in range(9)),[])
    y.extend([y[-1],]*(len(X)-len(y)))
    y = np.array(y, dtype=np.uint8)
    return X,y


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    X,y = make_9_3D_blobs()

    sp = plt.subplot(111, projection='3d')
    sp.scatter(*X.T, c=y)
    plt.show()

if __name__=='__main__':main()

















