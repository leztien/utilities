
def make_3D_S_shape_out_of_2D_data(data):
    import numpy as np
    assert isinstance(data, np.ndarray) and data.ndim == 2, "wrong input data"
    X = data
    t = (X[:,0] - X[:,0].mean()) * np.pi * 3
    x = np.sin(t)
    y = (np.cos(t)-1) * np.sign(t)
    z = X[:,-1]
    X = np.vstack([x,y,z]).T
    # rotate and scale
    n = 1 / (2 ** 0.5)
    T = [[n, -n, -n],
         [n, n, -n],
         [n, 0, n]]
    #X = np.matmul(T, X.T).T
    X += np.abs(X.min(axis=0))
    return X




def main():
    import matplotlib.pyplot as plt, mpl_toolkits.mplot3d
    from myutils.datasets import draw_random_pixels_from_image #, make_3D_S_shape_out_of_2D_data

    X, y = draw_random_pixels_from_image("HELLO", n_points=1500)
    X_3D = make_3D_S_shape_out_of_2D_data(X)

    sp = plt.subplot(111, projection='3d')
    sp.scatter(*X_3D.T, c=y, cmap=plt.cm.viridis)
    sp.set(xlabel="x-axis", ylabel="y-axis", zlabel="z-axis")

    plt.show()
if __name__=="__main__":main()