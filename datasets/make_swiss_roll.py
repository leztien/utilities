
def make_swiss_roll(n_revolutions=3, radius_to_width_ratio=1, major_minor_axis_ratio=1, density=300):
    """makes Swiss Roll"""
    import numpy as np
    from pandas import factorize
    from math import pi as π

    n_points = density    # relative data density
    t = np.linspace(0, n_revolutions * π, n_points)
    x = np.cos(t) * np.exp(0.1 * t) * major_minor_axis_ratio
    y = np.sin(t) * np.exp(0.1 * t)

    # calculate how many points to skip because of the exponential distance growth to make the distances equal
    M = np.c_[x, y]
    distances_between_points = ((M[1:, :] - M[:-1, :]) ** 2).sum(axis=1) ** 0.5
    length = len(distances_between_points)
    mult = length / np.max(distances_between_points - np.min(distances_between_points))
    nx = (distances_between_points - np.min(distances_between_points)) * mult
    nx = [int(n) for n in nx[::3]]
    x = x[::-1][nx][::-1]
    y = y[::-1][nx][::-1]
    t = t[::-1][nx][::-1]

    # calculate the step (for the distance between points) along the width of the roll
    M = np.c_[x, y]
    step = np.mean(((M[1:, :] - M[:-1, :]) ** 2).sum(axis=1) ** 0.5) * 1.5
    mn, mx = np.min(np.c_[x, y]), np.max(np.c_[x, y])
    z = np.arange(mn, mx * radius_to_width_ratio, step=step, dtype=np.float32)

    # assemble the matrix
    pl = np.zeros(shape=(len(z), len(x), 3), dtype=np.float64)
    pl[:, :, 0] = x
    pl[:, :, 1] = y

    pl = np.rot90(pl, axes=(0, 1), k=1)
    pl[:, :, -1] = z
    pl = np.rot90(pl, axes=(0, 1), k=-1)
    X = pl.reshape(pl.shape[0] * pl.shape[1], pl.shape[-1])

    # rotate and scale
    n = 1 / (2 ** 0.5)
    T = [[n, -n, -n],
         [n, n, -n],
         [n, 0, n]]
    X = np.matmul(T, X.T).T
    X += np.abs(X.min(axis=0))

    y = factorize(list(t) * len(z))[0].tolist()
    return X ,y





def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, y = make_swiss_roll(density=220, n_revolutions=3)

    fig = plt.figure(figsize=(10, 5))
    sp = fig.add_subplot(121, projection='3d')
    sp.axis('equal')
    sp.set(xlabel="x-axis", ylabel="y-axis", zlabel="z-axis")
    sp.scatter(*X.T, c=y, cmap='viridis', s=5)

    # plot different manifold results
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS
    model_list = (Isomap, LocallyLinearEmbedding, TSNE, MDS)

    for i, spNumber in enumerate([3, 4, 7, 8]):
        X_2D = model_list[i](n_components=2).fit_transform(X, y)

        sp = fig.add_subplot(2, 4, spNumber)
        sp.scatter(*X_2D.T, c=y, cmap='viridis', s=10)
        sp.set_xticks([]);
        sp.set_yticks([])
        sp.set_title(model_list[i].__name__, fontsize=10)

    plt.subplots_adjust(left=0.01, wspace=0.2)
    plt.show()

if __name__ == "__main__": main()