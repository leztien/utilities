

def entropy(a):
    assert hasattr(a, '__iter__'), "bad argument"
    from math import log2 as log
    a = list(a) if not isinstance(a, list) else a
    unique_values = frozenset(a)
    n = len(a)
    pp = (a.count(v)/n for v in unique_values)
    E = -sum(p * log(p) for p in pp)
    return E

def information_gain(a,f): # a = discreet vector; f = factor vector to partition a on
    assert all(hasattr(o, '__iter__') for o in (a,f)) and len(a)==len(f), "bad arguments"
    n = len(a)
    E = entropy(a)
    partitions = ([x1 for x1,x2 in zip(a,f) if x2==v] for v in set(f))
    weighted_average_of_entropies_partitioned = sum(entropy(a)*(len(a)/n) for a in partitions)
    I = E - weighted_average_of_entropies_partitioned
    return(I)



def main():
    import scipy.stats
    H = scipy.stats.entropy([0.5, 0.3, 0.2], base=2)
    a = (1,2,3,1,2,3,3)
    f = (0,1,0,1,0,1,1)
    E = entropy(a)              ;print(E)
    I = information_gain(a,f)   ;print(I)

if __name__=='__main__':main()


