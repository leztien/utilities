


def nearest_points_to_grid(data, nrows=10, sparseness=1, return_grid_dots=False):
    def pairwise_distance(data1: 'distance from these points',
                          data2: 'distance to these points',
                          maxdistance=None, nrows=10, sparseness=1):
        from numpy import ndarray, matrix, unique
        if not all(isinstance(data, ndarray) or isinstance(data, matrix) for data in (data1, data2)): raise TypeError(
            "The data must be ndarrays")
        mx = ((data1[None, :, :] - data2[:, None, :]) ** 2).sum(axis=-1)
        mask = mx > ((maxdistance / sparseness) ** 2)
        mx[mask] = mx.max()
        nx = mx.argmin(axis=0)
        nx = unique(nx)
        return nx

    from numpy import linspace, arange, meshgrid, c_
    a = data[:, 1]
    yy, step = linspace(start=a.min(), stop=a.max(), num=nrows + 1, endpoint=True, retstep=True)
    yy = (yy + step / 2)[:-1]

    a = data[:, 0]
    xx = arange(start=a.min(), stop=a.max(), step=step)
    offset = (a.max() - xx[-1]) / 2
    xx += offset

    xx, yy = meshgrid(xx, yy)
    grid_dots = c_[xx.ravel(), yy.ravel()]

    nx = pairwise_distance(grid_dots, data, maxdistance=step, nrows=nrows, sparseness=sparseness)
    return nx if not return_grid_dots else (nx, grid_dots)


#############################################################################

def main():
    from matplotlib.pyplot import subplots, show
    from numpy.random import multivariate_normal
    cm = [[4, 0],
          [0, 2]]

    blob = multivariate_normal(mean=[10, 10], cov=cm, size=500)
    nrows = 20
    nx, grid = nearest_points_to_grid(blob, nrows=nrows, sparseness=1, return_grid_dots=True)

    fig, sp = subplots(subplot_kw={'aspect': 'equal'})

    sp.scatter(*blob.T, s=5)
    sp.scatter(*grid.T, s=1, color='red', zorder=-2)
    sp.scatter(*blob[nx].T, color='k', s=3)
    show()

if __name__=='__main__':main()







