
# visualize cost function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def cost_function(slope, xx, yy):
    assert all('ndarray' in str(type(e)) for e in (xx,yy)), "arguments must be ndarrays"
    mse = (( yy.mean() - yy - slope*(xx.mean() - xx))**2).sum() / len(xx)
    return mse

def draw_cost_function(xx,yy, ax=None):
    from matplotlib.pyplot import gca
    from numpy import vectorize, tan, deg2rad

    sp = ax or gca()
    slopes_batch1 = tan(deg2rad(range(90)))
    slopes_batch2 = tan(deg2rad(range(91,180)))

    from functools import partial
    func = partial(cost_function, xx=xx, yy=yy)
    cost_function_vectorized = vectorize(func)

    mse_batch1 = cost_function_vectorized(slopes_batch1)
    mse_batch2 = cost_function_vectorized(slopes_batch2)

    sp.plot(slopes_batch1, mse_batch1)
    sp.plot(slopes_batch2, mse_batch2)
    sp.axvline(0, color='gray', linewidth=0.5)
    sp.axhline(0, color='gray', linewidth=0.5)
    sp.set_ylim([-1,10])
    sp.set_xlim([-5,5])
    sp.set_title('cost function')
    return sp


def gradient_descent(xx,yy):
    from numpy import vectorize, tan, deg2rad, argmin
    slopes = tan(deg2rad(tuple(range(90))+tuple(range(91,180))))

    from functools import partial
    func = partial(cost_function, xx=xx, yy=yy)
    cost_function_vectorized = vectorize(func)
    mse = cost_function_vectorized(slopes)
    nx = argmin(mse)
    slope = slopes[nx]
    intercept = yy.mean() - slope*xx.mean()
    return slope, intercept





def main():
    fig = plt.figure()
    sp = fig.add_subplot(121)


    mxCov = [[2, 2],
             [2, 4]]
    M = np.random.multivariate_normal(mean=[10, 15], cov=mxCov, size=100)
    xx, yy = M.T
    sp.scatter(xx,yy, marker='.')
    slope, intercept, *_ = linregress(xx,yy)
    f = lambda x : slope*x + intercept
    n = 100
    sp.plot([-n,n], [f(-n), f(n)], color='red', linewidth=1.5)
    sp.set_ylim([0, max(yy)])
    sp.set_xlim([0, max(xx)])
    sp.set_title('regression line')

    mxCov = np.cov(*M.T, ddof=0)
    print(mxCov)

    o = cost_function(1, xx,yy)
    print(o)



    slope, intercept = gradient_descent(xx,yy)
    f2 = lambda x : slope*x + intercept
    n = 100
    sp.plot([-n,n], [f2(-n), f2(n)], color='green', linewidth=1.5, linestyle=':')



    sp = fig.add_subplot(122)
    draw_cost_function(xx,yy, sp)



    plt.show()
if __name__=='__main__':main()