
class Regression3DByGradientDescent():
    def __init__(self):
        self.X = None
        self.xx, self.yy, self.zz = [None,]*3
        self.w0 = None
        self.w1 = None
        self.w2 = None

    def _weights_from_slopes(self, w1,w2, slope_units='deg'):
        from math import tan, radians
        # slope_units={'tangent', 'degrees', 'radians'}
        slope_units = slope_units[:3].lower()
        xbar, ybar, zbar = self.X.mean(axis=0)
        if slope_units == 'tan':
            pass
        elif slope_units == 'deg':
            w1,w2 = (tan(radians(e)) for e in (w1,w2))
        elif slope_units == 'rad':
            w1,w2 = (tan(e) for e in (w1,w2))
        else:
            raise ValueError("slope units not determined!")
        w0 = zbar - w1*xbar - w2*ybar
        return (w0,w1,w2)

    @staticmethod
    def distance_from_point_to_plane(x,y,z, *args):
        """A,B,C,D  must be provided from this form Ax+By+Cz+D=0    (mind the sign of D)"""
        from numpy import meshgrid
        from math import sqrt
        assert 4 >= len(args) >= 3, "either 4 or 3 arguments are needed"
        if len(args)==3:
            w0, w1, w2 = args
            a,b,c,d = -w1, -w2, 1, -w0
        elif len(args)==4:
            a,b,c,d = args
        else: raise Exception("unforseen error")
        d = abs(x*a + y*b + z*c + d) / sqrt(a**2 + b**2 + c**2)    #https://mathinsight.org/distance_point_plane
        return d

    def _mod(self, angle1, angle2): #angles must be in degrees
        """mean orthoganal distance"""
        from numpy import vectorize
        w0, w1, w2 = self._weights_from_slopes(angle1, angle2, slope_units='deg')
        fv = vectorize(self.distance_from_point_to_plane)
        sod = fv(self.xx, self.yy, self.zz, w0,w1,w2).sum() / len(self.xx)
        return (sod, w0, w1, w2)

    def _mse(self, angle1, angle2): # returns a tuple of 3 values
        """mean sum of squared errors = mean squared error"""
        from numpy import vectorize
        from functools import partial
        w0, w1, w2 = self._weights_from_slopes(angle1, angle2, slope_units='deg')
        yy_true = self.zz

        def func(x,y, w0,w1,w2):
            return x*w1 + y*w2 + w0
        fp = partial(func, w0=w0, w1=w1, w2=w2)
        fv = vectorize(fp)

        yy_pred = fv(x=self.xx, y=self.yy)
        SSErr = ((yy_true - yy_pred) ** 2).sum() / len(self.xx)
        return (SSErr, w0, w1, w2)


    def fit(self, X):
        from numpy import zeros, argmin, argsort
        from itertools import product
        from collections import deque
        self.X = X
        self.xx, self.yy, self.zz = self.X.T

        table = zeros(shape=(4, 6), dtype='f')
        """table columns: mse/mod     angle1  angle2  w0  w1  w2"""

        g = product([45,135],[45,135])
        for i,t in enumerate(g):
            table[i,:2] = t
            metric,w0,w1,w2 = self._mse(*t)     #mse shows better results
            table[i,2:] = metric,w0,w1,w2

        d = {45: (1, 89), 135: (91, 179), 'angle0':table[argmin(table[:,2]),0], 'angle1':table[argmin(table[:,2]),1]}
        table = zeros(shape=(3, 6), dtype='f')

        """LOOP"""
        for i in (0,1):     #column 0 or column 1
            table = zeros(shape=(3, 6), dtype='f')
            table[:2, i] = d.get( d.get(f'angle{i}', 45), (1,179) )
            [table.__setitem__((j, slice(2, None)), self._mod(*table[j, :2])) for j in (0, 1)]

            qu = deque(maxlen=3)
            for _ in range(100):     #max no. of loops- can be more
                table[-1, i] = table[:2, i].mean()
                table[-1, 2:] = self._mod(*table[-1, :2])
                table = table[argsort(table[:,2],axis=0)]
                qu.append(table[0,i])
                if len(qu)==3 and len(set(qu))==1:
                    d['angle{}'.format(i)] = table[0,i]
                    break

        """FINE TUNING"""
        for i in (0,1):     #column 0 or column 1
            table = zeros(shape=6, dtype='f')
            table[i] = d[f'angle{i}']
            table[2:] = self._mse(*table[:2])

            increment = 1
            bDirectionAlreadyChanged=False
            while True:
                angle1, angle2 = table[:2]
                mse,w0,w1,w2 = self._mse(angle1 + (increment if i==0 else 0), angle2 + (increment if i==1 else 0))
                if mse < table[2]:
                    table[i] += increment
                    table[2:] = mse,w0,w1,w2
                    continue
                elif mse > table[2]:
                    table[i] += increment
                    table[2:] = mse,w0,w1,w2
                    if bDirectionAlreadyChanged:
                        d['angle{}'.format(i)] = table[i]
                        break
                    increment = -increment
                    bDirectionAlreadyChanged=True
                    continue
                elif mse == table[2]:
                    table[i] += increment
                    table[2:] = mse,w0,w1,w2
                    d['angle{}'.format(i)] = table[i]
                    break
                else: raise Exception("unforseen error")
        self.w0, self.w1, self.w2 = self._weights_from_slopes(d['angle0'], d['angle1'])
        return self

    def fit2(self, X):
        from numpy import zeros, argmin
        from itertools import product
        from math import inf

        self.X = X
        self.xx, self.yy, self.zz = self.X.T

        table = zeros(shape=(4, 6), dtype='f')
        """table columns: mse/mod     angle1  angle2  w0  w1  w2"""

        g = product([45,135],[45,135])
        for i,t in enumerate(g):
            table[i,:2] = t
            metric,w0,w1,w2 = self._mse(*t)     #mse shows better results
            table[i,2:] = metric,w0,w1,w2

        d = {45: (1, 90), 135: (91, 180), 'angle0':table[argmin(table[:,2]),0], 'angle1':table[argmin(table[:,2]),1]}

        presision = 1
        angle1_range = range(*d.get(table[argmin(table[:, 2]), 0], (1,179)), presision)
        angle2_range = range(*d.get(table[argmin(table[:, 2]), 1], (1, 179)), presision)
        table = zeros(shape=6, dtype='f')
        table[2] = inf

        g = product(angle1_range, angle2_range)
        for t in g:
            mse = self._mse(*t)
            if mse[0] < table[2]:
                table = [*t, *mse]
        self.w0, self.w1, self.w2 = table[3:]
        return self


#=================================================================================

class Make3DDataset():
    def __init__(self, means=None, covariance_matrix=None):
        from sklearn.linear_model import LinearRegression
        from numpy import cov, meshgrid
        from numpy.random import multivariate_normal
        from matplotlib.pyplot import gca, figure
        from mpl_toolkits.mplot3d import Axes3D
        self.sp = Axes3D(figure())

        means = means or [10,15,20]
        covariance_matrix = covariance_matrix or [[1,1,1],
                                                  [1,2,3],
                                                  [1,3,7]]
        self.X = multivariate_normal(mean=means, cov=covariance_matrix, size=100)
        self._covariance_matrix = cov(self.X.T)

        self.sp.scatter(*self.X.T, marker='.', color='k')
        self.xbar, self.ybar, self.zbar = self.X.mean(axis=0)
        self.sp.scatter(self.xbar, self.ybar, self.zbar, marker='o', color='red', s=100)
        md = LinearRegression()
        md = md.fit(self.X[:,:2], self.X[:,-1][:,None])
        w1,w2 = md.coef_[0]
        w0 = md.intercept_[0]

        X,Y = meshgrid(sorted(self.X[:,0])[::len(self.X[:,0])-1], sorted(self.X[:,1])[::len(self.X[:,1])-1])
        Z = X*w1 + Y*w2 + w0
        self.sp.plot_surface(X,Y,Z, color='gray', alpha=0.2)
        self.sp.plot_wireframe(X, Y, Z, color='blue', alpha=0.6)
        self.sp.set_xlabel('x-axis');self.sp.set_ylabel('y-axis');self.sp.set_zlabel('z-axis')
        mn,mx = sorted(self.X.ravel())[::len(self.X.ravel())-1]
        #self.sp.set_xlim([mn,mx]);self.sp.set_ylim([mn,mx]);self.sp.set_zlim([mn,mx])
        #self.sp.axis('equal')
        self.w0,self.w1,self.w2 = w0,w1,w2

    @property
    def covariance_matrix(self):
        return self._covariance_matrix

    def add_plane(self, *args, **kwargs):
        """A,B,C,D  must be provided from this form Ax+By+Cz+D=0 (mind the sign of D)  OR use this form: w0,w1,w2"""
        assert 4 >= len(args) >= 3, "either 4 or 3 arguments are needed"
        from numpy import meshgrid
        X, Y = meshgrid(sorted(self.X[:, 0])[::len(self.X[:, 0]) - 1], sorted(self.X[:, 1])[::len(self.X[:, 1]) - 1])
        if len(args)==3:
            w0, w1, w2 = args
            Z = X * w1 + Y * w2 + w0
        elif len(args)==4:
            a,b,c,d = args
            Z = (X*-a + Y*-b - d) / c           # the sign of d is still in question
        else: raise Exception("unforseen error")
        color = kwargs.get('color', 'lightblue')
        self.sp.plot_surface(X, Y, Z, color=color, alpha=0.6)
        return self.sp

    def show(self):
        from matplotlib.pyplot import show
        show()


####################################################################################################



def main():
    ds = Make3DDataset()
    ##print(ds.covariance_matrix)
    X = ds.X
    xbar,ybar,zbar = X.mean(axis=0)

    ds.add_plane(ds.w0, ds.w1, ds.w2)
    #ds.add_plane(-ds.w1, -ds.w2, 1, -ds.w0)  #same as the preceeding line

    md = Regression3DByGradientDescent()
    md = md.fit(X)
    ds.add_plane(md.w0, md.w1, md.w2, color='yellow')


    md = md.fit2(X)
    ds.add_plane(md.w0, md.w1, md.w2, color='green')

    ds.show()
if __name__=='__main__':main()













