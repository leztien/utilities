from .productivity import unravel, caching_decorator_function_fast, caching_decorator_function, clocking_decorator_function
from .mapables import DictionaryForUnhashables
from .nearest_points_to_grid import nearest_points_to_grid
from.compare_console_output_of_two_functions import compare_console_output_of_two_functions
from.structured_tuple_of_ordered_categorical_variables import Type, Tuple, sort_categorical_tuples
__version__ = "version: n/a"




def transpose(mx, copy=True):
    assert len(mx)>1 and len(set(len(a) for a in mx))==1, "bad args"
    m,n = len(mx), len(mx[0])
    new = [[0,]*m for _ in range(n)]
    [new[j].__setitem__(i, mx[i][j]) for i in range(m) for j in range(n)]
    if copy==False:
        mx[:] = new[:]
        return mx
    return new

def vectorize(func):
    def _func(mx, copy=True):
        from numpy import ndarray, matrix
        if type(mx) in (ndarray, matrix):
            mx = mx.tolist()
        assert isinstance(mx,list), "must be a list"
        mx = transpose(mx, copy)
        for a in mx:
            a[:] = func(a)
        mx = transpose(mx, copy)
        return mx
    return _func

@vectorize
def scale(a):
    mn,mx = min(a),max(a)
    r = mx-mn
    a = [(x-mn)/r for x in a]
    return a

def split(mx):
    global transpose
    mx = transpose(mx)
    X,y = transpose(mx[:-1]), mx[-1]
    return X,y


