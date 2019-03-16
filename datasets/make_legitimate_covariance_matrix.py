




def make_legitimate_covariance_matrix(ndim=3, lo=2, hi=15) -> list:
    from numpy import zeros, int16
    from numpy.random import multivariate_normal, randint
    from numpy import warnings
    n = ndim
    for _ in range(100000):
        mx = zeros(shape=(n, n), dtype=int16)
        for i in range(n):
            for j in range(i, n):
                mx[i, j:] = randint(lo, hi, size=len(mx[i, j:]))
        cm = mx | mx.transpose()

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                mx = multivariate_normal(mean=[0,]*n, cov=cm, size=200)
                print(cm)
                return cm.tolist()
                break
            except RuntimeWarning:
                continue
    else:
        from warnings import warn
        warn("failed to find a covariance matrix. try again", Warning)
        return None

def main():
    cm = make_legitimate_covariance_matrix(4, -15,15)
    print(cm)
if __name__=='__main__':main()



