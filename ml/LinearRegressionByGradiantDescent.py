
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
#np.random.seed(0)

class LinearRegressionByGradiantDescent():
    def __init__(self, xx, yy):
        from numpy import array
        assert all(hasattr(e, '__iter__') for e in (xx,yy)), "the arguments must be iterable"
        assert len(xx)==len(yy), "the arrays must be of equal length"
        self.xx, self.yy = (xx,yy) if all('ndarray' in str(type(e)) for e in (xx,yy)) else [array(e) for e in (xx,yy)]

    @staticmethod
    def slope2degrees(slope):
        from math import atan, degrees
        angle_in_degrees = abs(degrees(atan(slope)))
        return angle_in_degrees

    @staticmethod
    def degrees2slope(angle_in_degrees):
        from math import tan, radians
        return tan(radians(angle_in_degrees))

    def intercept_from_slope(self, slope=0, slope_units='tan', return_slope_and_intercept=False):
        from math import tan, radians
        # slope_units={'tangent', 'degrees', 'radians'}
        slope_units = slope_units[:3].lower()
        x, y = self.xx.mean(), self.yy.mean()
        if slope_units == 'tan':
            pass
        elif slope_units == 'deg':
            slope = tan(radians(slope))
        elif slope_units == 'rad':
            slope = tan(slope)
        else:
            raise ValueError("slope units not determined!")
        intercept = y - slope * x
        return (slope, intercept) if return_slope_and_intercept else intercept      #the slope returned is in tan units

    @classmethod
    def orthogonal_distance(cls, x,y, slope, intercept):
        from math import sin, radians
        slope_in_degrees = cls.slope2degrees(slope)
        orth_dict = abs(y - (x*slope+intercept)) * sin(radians(90-slope_in_degrees))
        return orth_dict

    def sum_of_orthogonal_distances(self, slope, intercept):
        from numpy import vectorize
        from functools import partial
        f = partial(self.orthogonal_distance, slope=slope, intercept=intercept)
        fv = vectorize(f)
        return fv(self.xx, self.yy).sum()

    def sum_of_orthogonal_distances_from_angle(self, angle_in_degrees): #wrapper function
        slope, intercept = self.intercept_from_slope(slope=angle_in_degrees, slope_units='deg', return_slope_and_intercept=True)
        sod = self.sum_of_orthogonal_distances(slope, intercept)
        return (sod, slope, intercept)

    def mse(self, angle_in_degrees): # returns a tuple of 3 values
        from numpy import vectorize
        from functools import partial
        slope, intercept = self.intercept_from_slope(slope=angle_in_degrees, slope_units='deg', return_slope_and_intercept=True)
        yy_true = self.yy

        def func(x, slope, intercept):
            y = x*slope+intercept
            return y
        fp = partial(func, slope=slope, intercept=intercept)
        fv = vectorize(fp)

        yy_pred = fv(self.xx)
        SSErr = ((yy_true - yy_pred) ** 2).sum() / len(self.xx)
        return (SSErr, slope, intercept)

    def gradient_descent(self):
        from numpy import zeros
        table = zeros(shape=(3, 4), dtype='f')
        t = (0, 89) if self.sum_of_orthogonal_distances_from_angle(45) < self.sum_of_orthogonal_distances_from_angle(135) else (89, 179)
        table[:2, 0] = t
        [table.__setitem__((i, slice(1, None)), self.sum_of_orthogonal_distances_from_angle(table[i, 0])) for i in (0, 1)]

        # LOOP
        sod = table[0,1]
        for i in range(8):
            table[-1, 0] = table[:2, 0].mean()
            table[-1, 1:] = self.sum_of_orthogonal_distances_from_angle(table[-1, 0])
            table = table[np.argsort(table[:, 1], axis=0)]
            if sod == table[0,1]: break
            else: sod = table[0,1]
        else: raise Exception("exceeded number of loops")
        counter = i+2

        #fine tuning
        table = table[0]
        table[1] = self.mse(table[0])[0]
        increment = 1
        bDirectionAlreadyChanged=False
        while True:
            counter += 1
            sod, slope, intercept = self.mse(table[0] + increment)
            if sod < table[1]:
                table[:] = (table[0]+increment, sod, slope, intercept)
                continue
            elif sod > table[1]:
                if bDirectionAlreadyChanged:
                    break
                increment = -increment
                bDirectionAlreadyChanged=True
                continue
            elif sod == table[1]:
                table[:] = (table[0] + increment, sod, slope, intercept)
                break
            else: raise Exception("unforseen error")

        slope, intercept = table[2:]
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.info("zerod in in {} loops".format(counter))
        return (slope, intercept)

##############################################################################################

def main():
    mxCov = [[2, 3],
             [3, 25]]
    M = np.random.multivariate_normal(mean=[10, 15], cov=mxCov, size=100)
    xx, yy = M.T
    x, y = M.mean(axis=0)

    #true regression line
    slope, intercept, *_ = linregress(xx,yy)
    true_angle = LinearRegressionByGradiantDescent.slope2degrees(slope)
    print("true angle:", true_angle)
    print("true slope, intercept:", slope, intercept)

    f1 = lambda x : x*slope + intercept
    sp = plt.axes()
    sp.scatter(*M.mean(axis=0), color='red', marker='o')
    sp.scatter(xx,yy, marker='.')
    sp.plot([-100,100],[f1(-100),f1(100)], color='red', linewidth=1.5)
    sp.axis([0,max(M.ravel()),0,max(M.ravel())])

    ob = LinearRegressionByGradiantDescent(xx, yy)

    s,i = ob.gradient_descent()
    print("slope, intercept from my gradient descent", s,i)

    f3 = lambda x : x*s+i
    sp.plot([-100,100],[f3(-100),f3(100)], color='green', linewidth=1.5, linestyle='--')


    plt.show()
if __name__=='__main__':main()








