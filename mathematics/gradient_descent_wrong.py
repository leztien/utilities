from MyUtilities import print
import numpy as np, pandas as pd
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress

def gradient_descent(*args):
    from numpy import ndarray
    assert len(args)>0, "didn't find any arguments"
    assert all(hasattr(e, "__iter__") for e in args), "The argument(s) must be iterable."
    assert all(isinstance(e, ndarray) for e in args), "The argument(s) must be numpy ndarray(s)."
    if len(args)==1:
        xx,yy = args[0].T
    elif len(args)==2:
        xx,yy = args
    else: raise TypeError("wrong arguments format")
    assert all(hasattr(e, '__iter__') for e in (xx,yy)), "xx and yy must be iterable"
    assert len(xx)==len(yy), "both arrays must be of equal length"

    xbar,ybar = xx.mean(), yy.mean()
    table = np.zeros(shape=(3,4), dtype=np.float_)

    def mse_from_angle(slope_in_degrees):
        def intercept_from_slope(point=(xbar,ybar), slope=slope_in_degrees, slope_units='deg', return_intercept_and_slope=True):
            from math import tan, radians
            # slope_units={'tangence', 'degrees', 'radians'}
            slope_units = slope_units[:3].lower()
            x, y = point
            if slope_units == 'tan':
                pass
            elif slope_units == 'deg':
                slope = tan(radians(slope))
            elif slope_units == 'rad':
                slope = tan(slope)
            else:
                raise ValueError("slope units not determined!")
            intercept = y - slope * x
            return (intercept, slope) if return_intercept_and_slope else intercept

        def line_function(intercept, slope):
            from numpy import vectorize
            @vectorize
            def _func(x):
                y = intercept + x * slope
                return y
            return _func

        def mse(func):  # mean_sum_of_squared_errors
            assert all(hasattr(e, '__iter__') for e in (xx, yy)), "xx and yy must be iterable"
            assert len(xx) == len(yy), "both arrays must be of equal length"
            from numpy.lib.function_base import vectorize
            assert func.__class__ is vectorize, "provided function must be a numpy vectorized function"
            from numpy import array
            yy_true = array(yy)
            yy_pred = func(xx)
            SSErr = ((yy_true - yy_pred) ** 2).sum() / len(xx)
            return SSErr

        intercept, slope = intercept_from_slope(point=(xbar,ybar), slope=slope_in_degrees)
        f = line_function(intercept, slope)
        SSErr = mse(f)
        return (SSErr, intercept, slope)
    #----------------------------------------------------------------------------------



    t = (0,89) if mse_from_angle(45)[0] < mse_from_angle(135)[0] else (89,179)
    table[:2,0] = t
    [table.__setitem__((i,slice(1,None)), mse_from_angle(table[i,0])) for i in (0,1)]


    #LOOP
    for i in range(8):
        table[-1,0] = table[:2,0].mean()
        table[-1,1:] = mse_from_angle(table[-1,0])
        table = table[np.argsort(table[:,1], axis=0)]
        print('step',i,table)


    #select the best answer
    intercept, slope = table[0,2:]
    return (slope, intercept)
#====END OF function======================================================================================


def intercept_from_slope(point=(1, 1), slope=0, slope_units='tan', return_intercept_and_slope=False):
    from math import tan, radians
    # slope_units={'tangence', 'degrees', 'radians'}
    slope_units = slope_units[:3].lower()
    x, y = point
    if slope_units == 'tan':
        pass
    elif slope_units == 'deg':
        slope = tan(radians(slope))
    elif slope_units == 'rad':
        slope = tan(slope)
    else: raise ValueError("slope units not determined!")
    intercept = y - slope * x
    return (intercept, slope) if return_intercept_and_slope else intercept

def line_function(intercept, slope):
    from numpy import vectorize
    @vectorize
    def _func(x):
        y = intercept + x*slope
        return y
    return _func

def mse(xx, yy, func):       #mean_sum_of_squared_errors
    assert all(hasattr(e, '__iter__') for e in (xx,yy)), "xx and yy must be iterable"
    assert len(xx)==len(yy), "both arrays must be of equal length"
    from numpy.lib.function_base import vectorize
    assert func.__class__ is vectorize, "provided function must be a numpy vectorized function"
    from numpy import array
    yy_true = array(yy)
    yy_pred = func(xx)
    SSErr = ((yy_true - yy_pred)**2).sum() / len(xx)
    return SSErr













def main():
    sp = Axes3D(plt.figure())
    sp = plt.axes()

    mxCov = [[2, 3],
             [3, 5]]
    M = np.random.multivariate_normal(mean=[10, 15], cov=mxCov, size=100)
    xx,yy = M.T
    x, y = M.mean(axis=0)
    mxCov = np.cov(*M.T, ddof=0)
    print(mxCov)

    sp.scatter(*M.T, marker='.')
    sp.scatter(x,y, marker='o', color='red')
    n = M.ravel().max()
    sp.axis([0, n, 0, n])

    #true regression line
    slope, intercept, *_ = linregress(xx,yy)
    n = 100
    f1 = lambda x : x*slope + intercept
    sp.plot([-n,n], [f1(-n), f1(n)], color='red', alpha=0.5)


    #regression line dieitermined by my function
    slope, intercept = gradient_descent(xx,yy)
    n = 100
    f2 = lambda x : x*slope + intercept
    sp.plot([-n,n], [f2(-n), f2(n)], color='blue', alpha=0.5)
    print(linregress(xx,yy))






    plt.show()


if __name__ == '__main__': main()








