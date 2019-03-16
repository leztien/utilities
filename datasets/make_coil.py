
def make_coil(n_revolutions=3, height_width_ratio = 1, strech_of_the_spring = 10, n_datapoints=100):
    """make a 3D spring/coil for decomposition with manifold algorithms.
    A 1D manifold embedded in 3D"""
    import numpy as np
    from math import pi as π
    n = n_revolutions  # how many revolutions/coils
    n_points_per_coil = n_datapoints / n_revolutions

    t = np.linspace(- π *n, π* n, n_points_per_coil * n)
    x = np.sin(t) * height_width_ratio  # flatten the circle
    y = np.cos(t)
    x, y, z = t / strech_of_the_spring, x, y
    M = np.vstack([x, y, z]).T

    n = 1 / (2 ** 0.5)
    T = [[n, -n, -n],
         [n, n, -n],
         [n, 0, n]]
    M = np.matmul(T, M.T).T
    M += np.abs(M.min(axis=0))
    X = M
    y = np.arange(1, len(M) + 1)
    return X, y


def make_tapering_coil(n_datapoints=100, n_revolutions=5, width_height_ratio=1, strech_of_the_spring=10):
    import numpy as np
    from math import pi as π
    t = np.linspace(0, n_revolutions * π, n_datapoints)
    x = np.cos(t) * np.exp(0.03 * t) * width_height_ratio
    y = np.sin(t) * np.exp(0.03 * t)

    x, y, z = t / strech_of_the_spring, x, y
    M = np.vstack([x, y, z]).T

    n = 1 / (2 ** 0.5)
    T = [[n, -n, -n],
         [n, n, -n],
         [n, 0, n]]
    M = np.matmul(T, M.T).T
    M += np.abs(M.min(axis=0))
    X = M
    y = np.arange(1, len(M) + 1)
    return X, y




def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, y = make_coil(n_revolutions=5, height_width_ratio=2, strech_of_the_spring=10, n_datapoints=1000)   # comment eithe line out
    X, y = make_tapering_coil(n_datapoints=1000, n_revolutions=10, width_height_ratio=1.5, strech_of_the_spring=10) # comment eithe line out


    fig = plt.figure(figsize=(10, 5))
    sp = fig.add_subplot(121, projection='3d')
    sp.axis('equal')
    sp.set(xlabel="x-axis", ylabel="y-axis", zlabel="z-axis")
    sp.scatter(*X.T, c=y, cmap='viridis', s=5)

    # plot different manifold results
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS
    model_list = (Isomap, LocallyLinearEmbedding, TSNE, MDS)

    sp1D = plt.axes([0.475, 0.01, 0.425, 0.1])

    for i, spNumber in enumerate([3, 4, 7, 8]):
        X_2D = model_list[i](n_components=2).fit_transform(X, y)
        X_1D = model_list[i](n_components=1).fit_transform(X, y)

        sp = fig.add_subplot(2, 4, spNumber)
        sp.scatter(*X_2D.T, c=y, cmap='viridis', s=10)
        sp.set_xticks([]);
        sp.set_yticks([])
        sp.set_title(model_list[i].__name__, fontsize=10)

        a = MinMaxScaler().fit_transform(X_1D)
        sp1D.scatter(a, [-i, ] * len(X_1D), c=y, cmap='viridis', s=10)

    sp1D.set_xlabel("data decomposed into 1D by the above models")
    sp1D.set_xticks([]);
    sp1D.set_yticks([])

    d = sp1D.spines
    [spine.set_color('none') for spine in d.values()]
    sp1D.set_frame_on(False)  # same effect as the preceeding line

    plt.subplots_adjust(left=0.01, wspace=0.2)
    plt.show()

if __name__ == "__main__": main()



















































