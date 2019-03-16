

#LOGGING FUNCTION
import logging
from inspect import stack
logging.basicConfig(level=logging.INFO, format='%(message)s')
def log(*args):
    s = stack()[1].function
    output = str(args[0] if len(args)==1 else str(args)[1:-1]).ljust(100) + "CALLER: " + s.strip("<>") + " | "
    logging.info(output)




#CLOCKING FUNCTION
from time import perf_counter
def clocking_decorator_function(func):       # func = free variable
    def closure_function(*args):
        t0 = perf_counter()
        output = func(*args)
        elapsed_time = perf_counter() - t0
        s = "elapsed time: {:.7f} sec.\t{:}({}) = {}".format(elapsed_time, func.__name__, str.join(", ", (repr(arg) for arg in args)), repr(output))
        print(s)
        return output
    return closure_function


#CACHING FUNCTION
from myutils.productivity.mapables import DictionaryForUnhashables
from inspect import signature
def caching_decorator_function(func):
    d = DictionaryForUnhashables()  # free-variable
    def closure_function(*args, **kwargs):
        nonlocal d
        # inspect.Signature & inspect.BoundArguments
        sig = signature(func)
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        args, kwargs = ba.args, ba.kwargs
        # get/set result
        result = d.get([args, kwargs], None) or d.setdefault([args, kwargs], func(*args, **kwargs))  # ba.args, ba.kwargs  are used in effect
        return result
    return closure_function


#A SLIGHTLY FASTER CACHING FUNCTION
from inspect import signature
from bisect import insort, bisect, bisect_right
def caching_decorator_function_fast(func):   # converts the args and kwargs into a string and then into a bytes-arrays and stores sorted in a list
    keys_list, values_list = [],[]
    def closure_function(*args, **kwargs):
        nonlocal keys_list, values_list
        sig = signature(func)
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        key = bytes(str(ba.args)+str(ba.kwargs), encoding='utf_8')

        if key in keys_list:
            result = values_list[keys_list.index(key)]
            return result
        else:
            result = func(*ba.args, **ba.kwargs)
            nx = bisect_right(keys_list, key)
            keys_list.insert(nx, key)
            values_list.insert(nx, result)
            return result
    return closure_function

#UNRAVEL A SEQUANCE
def unravel(it):
    l = list()
    for e in it:
        if hasattr(e, "__len__") and type(e) is not str: l.extend(unravel(e))
        else: l.append(e)
    return l

#######################################################################################################

def main():
    @caching_decorator_function
    def fn1(arg1, arg2=20, *args, kw1=100, kw2, kw3=300, **kwargs):
        return (arg1, arg2, args, kw1, kw2, kw3, kwargs)

    @caching_decorator_function
    def fn2(arg1, arg2=20, *, kw1, kw2, **kwargs):
        return (arg1, arg2, kw1, kw2, kwargs)

    @caching_decorator_function
    def factrorial(n):
        return 1 if n <= 1 else n * factrorial(n - 1)

    output = fn1(1, kw2=222)  # minimum required
    print("fn1 output:", output)

    output = fn1(1, kw2=222, kw4=444, kw5=555, arg2=22)
    print("fn1 output:", output)

    output = fn1(1, 2, 3, 4, 5, kw4=444, kw3=333, kw2=222, kw1=111)
    print("fn1 output:", output)

    output = fn1(1, 2, 3, 4, 5, kw4=444, kw3=333, kw2=222, kw1=111)  # repeated second time
    print("fn1 output:", output)

    output = fn2(1, kw1=111, kw2=222, kw3=333)
    print("fn2 output:", output)

    output = fn2(1, kw1=111, kw2=222, kw3=333)
    print("fn2 output:", output)

    log("testing the log function", output)

    l = [1, 2, 3, ['a', 'b'], [[[10], 20]], (((100, 200)), 300, "text", ["Abc", "Def"]), {11, 22, 33, (111, 222, 333, (11111, 22222, 33333))}]
    l = unravel(l)
    print(l)

    ################
    from time import perf_counter

    @caching_decorator_function_fast
    def fibonacci(n: "n'th number in the Fibonacci series") -> int:  # recursive function, returns n'th element of the Fibonacci series
        if n < 2: return n
        return fibonacci(n - 2) + fibonacci(n - 1)

    t0 = perf_counter()
    n = fibonacci(100)
    t1 = perf_counter() - t0
    print(n, t1)

    t0 = perf_counter()
    n = fibonacci(100)
    t1 = perf_counter() - t0
    print(n, float(t1))

if __name__ == "__main__": main()