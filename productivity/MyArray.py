

class Array():
    def __init__(self, *args):
        assert all(type(e) in (int,float) for e in args), "all elements must be numebrs"
        self._a = __import__('array', fromlist=('array')).array(*('f' if any(isinstance(e, float) for e in args) else 'i', tuple(args)))

    def __str__(self):
        return self._a.tolist().__str__().replace("[","< ").replace("]"," >")

    def __getitem__(self, i):
        return self._a[i]

    def __len__(self): return self._a.__len__()

    def __iter__(self):
        return self._a.__iter__()

    def _vectorized_operation(self, v):
        assert not hasattr(v, '__iter__') or len(v) in (len(self._a), 1), "unable to broadcast"
        s = __import__('inspect', fromlist=('stack')).stack()[1][3].replace("_","")
        arr = self._a
        values = v if hasattr(v, '__iter__') else [v, ] * len(self._a)
        if s[0] == 'r':
            arr, values = values, self._a
            s = s[1:]
        md = __import__('operator', fromlist=[s])
        func = eval('md.'+s)
        if len(values)!=len(self._a): values = [v[0],]*len(self._a)
        return self.__class__(*(func(e,v) for e,v in zip(arr, values)))

    def __add__(self, v): return self._vectorized_operation(v)
    def __sub__(self, v): return self._vectorized_operation(v)
    def __mul__(self, v):return self._vectorized_operation(v)
    def __truediv__(self, v):return self._vectorized_operation(v)
    def __pow__(self, v):return self._vectorized_operation(v)
    def __rsub__(self, v):return self._vectorized_operation(v)
    def sum(self): return sum(self._a)



def chi_square_test_for_array_of_counts(*args, confidence_level=None):
    if len(args)==1 and hasattr(args[0], '__iter__') and len(args[0])>=2: args = args[0]
    elif len(args) >= 2: pass
    else: raise Exception("wrong arguments")
    assert all(type(e) in (int,float) for e in args), "all elements must be int of float"
    assert all(isinstance(e, int) or float.is_integer(e) for e in args), "all numbers must be whole numbers"
    arr = Array(*args)
    mu = __import__('statistics', fromlist=('mean')).mean(args)
    chi2 = ((mu - arr)**2 / mu).sum()
    if confidence_level is None: return chi2
    if confidence_level > 1: confidence_level = confidence_level / 100
    assert 0 < confidence_level < 1, "confidence level must be a valid percentage"
    critical_value = __import__('scipy.stats', fromlist=('chi2')).chi2.ppf(confidence_level, len(arr)-1)
    nt = __import__('collections', fromlist=('namedtuple')).namedtuple("ChiSqTest", ['chi2','critical_chi2','reject_null_hypothesis'])(chi2, critical_value, chi2 > critical_value)
    return nt

def main():
    ans = chi_square_test_for_array_of_counts([30,29,32])
    print(ans)

    ans = chi_square_test_for_array_of_counts([111,90,81,102,124,92], confidence_level=95)
    print(ans)




if __name__=='__main__':main()