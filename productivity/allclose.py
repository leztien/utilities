def is_close(n1, n2, tolerance=0.001):
    from numbers import Real
    assert all(isinstance(n, Real) for n in (n1,n2)), "bad input"
    n1,n2 = sorted([n1,n2])
    b = round(1-n1/n2, 5) <= tolerance
    return b


def vectorizing_decorator(func): # for the is_close function
    return_mask = False     # default setting
    func = func             # free variable
    def closure_function(*args, **kwargs):
        if len(args)==3 and 'tolerance' not in kwargs:
            kwargs.update({'tolerance':args[-1]})
        elif len(args)!=2:
            raise ValueError("bad arguments")
        else: "ok"

        args = args[:2]
        from collections import Sequence
        s = str.join('', (str(int(isinstance(obj, Sequence))) for obj in (args)))

        if s=='00':
            result = func(*args, **kwargs)
            return result
        elif '0' in s:
            j = 1 if s=='01' else -1
            n,a = args[::j]
            n = [n,]*len(a)
            args = (n,a) if s=='01' else (a,n)
        else:
            pass
        assert len(args[0])==len(args[1]), "arrays of different size"
        mask = [None,]*len(args[0])
        for ix,(n1,n2) in enumerate(zip(*args)):
            mask[ix] = func(n1,n2, **kwargs)
        result = mask if return_mask else all(mask)
        return result
    return closure_function


all_close = vectorizing_decorator(is_close)