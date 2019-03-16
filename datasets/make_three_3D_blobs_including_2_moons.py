
def make_three_3D_blobs_including_2_moons(n_samples=1000):
    import numpy as np
    m = n_samples // 3

    #blob1
    Σ = [[16,10,6],
         [0,9,3],
         [0,0,4]]
    Σ = np.array(Σ, np.int8)
    Σ = Σ | Σ.T
    Σ = Σ / 40

    blob1 = np.random.multivariate_normal(mean=[-2,-2,-2], cov=Σ, size=m)

    T = [[0.8, 0, -1.4],
         [0, 0.7, -1.4],
         [0.5, 0.5, 0.5]]

    blob1 = np.matmul(T, blob1.T).T


    #blob2
    t = np.linspace(0, np.pi, m)

    x = t
    y = np.sin(x)
    z = np.zeros_like(x)

    M = np.vstack([x,y,z]).T

    Σ = [[1,0,0],
         [0, 1, 0],
         [0, 0, 1]]

    ε = np.random.multivariate_normal(mean=[0,0,0], cov=Σ, size=m) / 5

    blob2 = M + ε

    #blob3
    x = np.full_like(t, fill_value=t.mean())
    y = -np.sin(t)
    z = t - t.mean()

    blob3 = np.vstack([x,y,z]).T + ε

    #combine blobs
    X = np.vstack([blob1, blob2, blob3])
    X += np.abs(X.min())
    y = sum(([n]*m for n in (0,1,2)), [])
    return X,y


def main():

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d


    sp = plt.subplot(111, projection='3d')

    X,y = make_three_3D_blobs_including_2_moons()

    sp.scatter(*X.T, c=y)

    sp.axis('equal')
    sp.set_xlabel("x");sp.set_ylabel("y");sp.set_zlabel("z")

    plt.show()

if __name__=='__main__':main()













