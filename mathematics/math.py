
from functools import lru_cache

@lru_cache(maxsize=128)
def factorial(n):
    result = 1 if n <= 1 else n * factorial(n - 1)
    return result

@lru_cache(maxsize=128)
def choose(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))


def binomial_expansion(degree=2, latex=False) -> "string in formula or latex format":
    assert (type(degree) is int) or (type(degree) is float and degree.is_integer()), "degree must be an integer"
    degree = int(degree)
    d = {str(i): eval("u'{}'".format(r'\u{}{}'.format('00B' if i in (1, 2, 3) else 207, 9 if i == 1 else i))) for i in range(10)}
    func = lambda n : '' if n==1 else ''.join(d[c] for c in str(n))
    coefficients = (choose(degree,k) for k in range(degree+1))
    exponents = list(range(degree+1))
    g = zip(coefficients, exponents[::-1], exponents)
    if latex:
        s = (' + '.join(("%sx^{%i}y^{%i}" % ('' if a == 1 else str(a), b, c)) for a, b, c in g)).replace("x^{0}", '').replace("y^{0}", '').replace("^{1}", '')
        s = ("(x + y)^{%i} = " % degree) + s
    else:
        s = (' + '.join("{}x{}y{}".format('' if a==1 else a, func(b), func(c)) for a,b,c in g)).replace("x⁰",'').replace("y⁰",'')
        s = "(x + y){} = ".format(func(degree)) + s
    return s

#=================================================================================================================================

def herz_frequency(of:int or str) -> float:
    from math import log as ln, exp
    if type(of) is str:
        s = of.lower()
        d = {k: v for k, v in zip("abcdefg", (0, 2, 3, 5, 7, 8, 10))}
        assert (1 <= len(s) <= 3) and (s[0] in d), "bad argument"
        if not s[-1].isdigit(): s += '4'
        x = sum([d[s[0]], 1 if s[1]=='#' else 0, (int(s[-1])-(4 if s[0] in "ab" else 5))*12])
    else:
        x = of
    xx,yy = (0,12), [ln(n) for n in (440,880)]
    slope = r = (yy[-1]-yy[0])/(xx[-1]-xx[0])
    i = exp(yy[0])  # intercept = initial number
    y = exp(x*r) * i
    return round(y,2)



def gaussian_kernel(data, landmark='mean', gamma=None):
    from numpy import array, exp
    X = data
    landmark = X.mean(axis=0) if landmark=='mean' else array(landmark)
    assert landmark.shape == (2,), "bad landmark"
    n = len(X)
    a = ((X-landmark)**2).sum(1)
    sqrt_a = a**0.5
    mu = sqrt_a.mean()
    sigma_squared = ((sqrt_a - mu)**2).sum() / (n-1)
    gamma = gamma or 1/(2 * sigma_squared)
    a = exp(-(gamma*a))
    return a


#########################################################################

def main():
    #DEMO
    s = binomial_expansion(degree=5, latex=True)
    print(s)
    s = binomial_expansion(degree=6)
    print(s)

    f = herz_frequency(0)
    f = herz_frequency(12)
    f = herz_frequency('A')     ;  print(f)
    f = herz_frequency('A4')
    f = herz_frequency('a#4')
    f = herz_frequency('C5')
    f = herz_frequency('C#0')
    f = herz_frequency(of='f#8')

if __name__ == '__main__':main()


