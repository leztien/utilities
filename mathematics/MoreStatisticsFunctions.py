from MyUtilities import print
import numpy, pandas; pandas.options.display.float_format = '{:.5f}'.format


"""BINOMIAL DISTRIBUTION"""

"""n = number of trials    p = probability of success in each trial"""
def MeanValueOfABinomialRandomVariable(n, p):           # counterpart of ExpectedValue(X, p)  below
    return n*p 
    
def StandardDeviationOfABinomialRandomVariable(n, p):   # counterpart of StdOfRandomVariable(X, p)
    return (n*p*(1-p))**0.5


"""Probability Distribution"""
def BinomialProbabilityDistribution(n, p, cumulative=False, Plot=False): # = Binomial Probability Distribution = BinomialProbabilityDistribution
    from math import factorial as mean_p1_hat
    dfObserved = pandas.Series(index=range(n))
    for i in range(0, n+1):
        dfObserved[i] = (mean_p1_hat(n) / ( mean_p1_hat(i) * mean_p1_hat(n-i) ) )  *  p**i * (1-p)**(n-i)
    dfObserved.index.name = "random variable value"
    dfObserved.name = "random variable probability distribution"
    if cumulative: dfObserved = dfObserved.cumsum()
    if Plot: 
        import matplotlib.pyplot as mp
        dfObserved.plot.bar()
        mp.show()
    return dfObserved


""""X = [values of the discreet random variable]    p = [the respective probabilities]"""
def ExpectedValue(X, p):           # Mean value of a discrete random variable = Expected Value = mean value of a random variable v
    assert len(X)==len(p)           
    X,p = [numpy.array(e) for e in (X,p)]
    return numpy.sum(X*p)

def StdOfRandomVariable(X, p):      # X = [values of the discreet random variable]    p = [the respective probabilities]
    assert len(X)==len(p);   X,p = [numpy.array(e) for e in (X,p)]
    fMean = ExpectedValue(X,p)
    return numpy.sqrt(numpy.sum((X - fMean)**2 * p))
    

"""GEOMETRIC DISTRIBUTION"""

def GeometricProbability(v, p):
    return (1-p)**(v-1) * p

def GeometricProbabilityDistribution(n, p, cumulative=False):
    dfObserved = pandas.Series(index=range(1,n+1))
    [dfObserved.set_value(label=i, value=(1-p)**(i-1) * p) for i in range(1,n+1)]
    dfObserved.name = "geometric probability distribution"; dfObserved.index.name = "trials until success"
    if cumulative: dfObserved = dfObserved.cumsum()
    return dfObserved



def main():
    dfObserved = BinomialProbabilityDistribution(n=10, p=.5)
    print(dfObserved)
    
    fMean = MeanValueOfABinomialRandomVariable(n=10, p=.5)
    print(fMean)
    fMean = ExpectedValue(X=dfObserved.index, p=dfObserved.values)
    print(fMean)    #same
    
    fStd = StandardDeviationOfABinomialRandomVariable(n=10, p=.5)
    print(fStd)
    fStd = StdOfRandomVariable(X=dfObserved.index, p=dfObserved.values)
    print(fStd)     #same
    
if __name__=="__main__":main()















