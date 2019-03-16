

def make_data(n=None, i=None, p=None):
    from random import randint
    from numpy import log, random as nprandom, arange as nparange, exp as npexp
    n = n or randint(10,100)
    i = i or randint(1,100)
    p = p or randint(1,50) / 100
    xx = nparange(1, n + 1)
    yy_clean = npexp(xx * p) * i
    yy = yy_clean + nprandom.normal(0, log(yy_clean).round() + (yy_clean//10), size=n)
    return (xx,yy,yy_clean, n,i,p)

def fit(xx,yy):
    from scipy.stats import linregress
    from numpy import log
    from math import exp
    yy_transformed = log(yy)
    slope, intercept, *_ = linregress(xx,yy_transformed)
    intercept = exp(intercept)
    return intercept, slope     #intercept = i,  slope = p

def predict_exponential_growth(xx, yy, n=10):
    from math import exp
    i, p = fit(xx,yy)
    xx = range(max(xx), max(xx)+n+1)
    yy = [exp(x*p)*i for x in xx]
    return (xx,yy)


def exponential_growth_prediction(values, ax=None, show_plot=True):
    assert \
        hasattr(values, '__iter__') and \
        hasattr(values, '__len__') and \
        len(values) > 2 and \
        not any(hasattr(e, '__len__') for e in values) and \
        all((type(e) in (int,float)) or ('numpy' in str(type(e))) for e in values), "wrong input"

    yy = values
    xx = range(1, len(yy)+1)

    #transform yy
    from math import log as ln, exp
    yy_transformed = [ln(y) for y in yy]

    #fit linear
    from scipy.stats import linregress
    slope,intercept = linregress(xx,yy_transformed)[:2]
    p=slope
    i=exp(intercept)

    #compose formula
    func = lambda x : i*exp(p*x)
    s = r"$y = %.2fe^{%.2fx}$" % (i,p)

    #predicted yy
    yy_pred = [func(x) for x in xx]

    #CAGR
    CAGR = yy_pred[1]/yy_pred[0]

    #plot
    from matplotlib.pyplot import gca, plot, scatter, show, subplot, legend
    sp = ax or subplot()

    #plot series lines and add to the legend
    legend(sp.plot(xx,yy,'.-r', xx,yy_pred,'.-b'), ["actual data","predicted growth"])

    sp.set_title("Exponential growth prediction")
    sp.text(0.1, 0.8, s, transform=sp.transAxes)
    sp.text(0.1, 0.75, r"$CAGR = %.1f%s$" % ((CAGR-1)*100, r"\%"), transform=sp.transAxes)

    if show_plot: show()

    #return
    from collections import namedtuple
    nt = namedtuple("exp_growth_prediction",
                    ['initial_ammount',
                     'exponent_multiplier',
                     'growth_percentage',
                     'latex_formula',
                     'prediction_function',
                     'subplot'])
    return nt(i, p, CAGR, s[1:-1], func, sp)


#========================================================================
def main():
    from matplotlib.pyplot import plot, gca, show, legend
    xx,yy,yy_clean, n,i,p = make_data()
    plot(xx,yy_clean, '-r', linewidth=0.5, alpha=0.5, label='original data w/out noise')
    plot(xx, yy, '.', label='data at hand')

    i, p = fit(xx,yy)
    print("intercept, slope:", i, p)

    gca().text(0.1, 0.6, '$y=%i e^{x %f}$' % (i,p), transform=gca().transAxes, fontweight='bold', fontsize=12)

    xx,yy = predict_exponential_growth(xx, yy)
    plot(xx,yy, ':y', label='prediction')

    legend(numpoints=3, fontsize=10)
    show()

    l = [112.2154984, 73.27502111, 114.1202231, 143.6907509, 187.2308737, 133.060562, 194.2783675, 312.279314, 319.0753193, 353.0477821, 348.24471, 476.5637427, 589.5352267, 579.383916, 705.0744758, 827.3978246, 919.9302995, 1051.102647, 1326.663586]

    nt = exponential_growth_prediction(l, show_plot=False)
    print(nt)

if __name__=='__main__':main()
