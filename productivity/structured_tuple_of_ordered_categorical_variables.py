

"""
structured tuple with predefined ordered categorical variables.
the tuples can be compared and sorted
e.g. Tuple(97, Dec, Wed) < Tuple(00, Aug, Thu)
"""


class Type:
    def __init__(self, *args, **kwargs):
        if len(args)==1: args=args[0]
        from collections.abc import Sequence
        if not all(isinstance(o, Sequence) and     # must be sequences
                   not isinstance(o,str) and       # but not strings
                   len(o)>1 and                    # of length greater than 1
                   hasattr(o, '__getitem__') and   # must be subscribable
                   len(o)==len(frozenset(o)) and   # must be all unique
                   all(isinstance(e, str) or not isinstance(e, Sequence) for e in o)
                   for o in args):
            raise TypeError("all must be subscribable iterables of length > 1 with unique values")

        self.cats = tuple(tuple(cat) for cat in args)   # unique categoricals
        key = ([key for key in kwargs.keys() if ('strict' in str(key).lower())] or ['nosuchkey'])[0]
        self.strict_eq = bool(kwargs.get(key, False))

    def __repr__(self):
        length = max(len(cat) for cat in self.cats)
        n = max(len(str(e)) for cat in self.cats for e in cat)
        cats = (tuple(str(e).ljust(n) for e in cat)+(' '.ljust(n),)*(length-len(cat)) for cat in self.cats)
        t = tuple(str.join(' | ', t) for t in zip(*cats))
        s = str.join('\n', t)
        s = "Type(\n" + s + " )"
        return s
    def __str__(self):
        return "{}(with {} categorical variables)".format(self.__class__.__name__, len(self))

    def __len__(self):
        return len(self.cats)

    def __eq__(self, other):
        if not isinstance(other, self.__class__): raise TypeError("Both must be of class %s"%self.__class__.__name__)
        from operator import eq, is_
        fn = is_ if (self.strict_eq or other.strict_eq) else eq
        b = (len(self)==len(other)) and\
            all(len(cat1)==len(cat2) for cat1,cat2 in zip(self,other)) and\
            all(fn(c1,c2) for cat1,cat2 in zip(self,other) for c1,c2 in zip(cat1,cat2))
        return b

    def __iter__(self):
        g = (cat for cat in self.cats)
        return g

    def conforms(self, tuple):
        t = tuple
        from collections.abc import Sequence
        if ((not isinstance(t, Sequence)) or isinstance(t, str)) or\
            any(isinstance(e, Sequence) and not isinstance(e,str) for e in t) or\
            len(t)<2:
            raise TypeError("must be a tuple of non-iterable values")

        b = (len(t)==len(self.cats)) and all(e in cat for e,cat in zip(t,self.cats))
        return b


class Tuple:   # structured tuple of ordered categorical variables
    """
    structured tuple of ordered categorical variables
    """
    def __init__(self, *components, type, from_type=False, update=False):
        if update==True:
            from warnings import warn
            warn("updating the Type with non-conformant values is not yet implemented", Warning)

        self.type = type
        from builtins import type
        if type(self.type) is not Type:
            self.type = Type(self.type)
            #raise TypeError("the type must be of class %s"%Type.__name__)
        if len(components)==1: components=components[0]
        if not(hasattr(components, '__iter__') or hasattr(components, '__len__')): raise TypeError("must have at least 2 components")
        from collections.abc import Sequence
        if any(isinstance(o, Sequence) and not isinstance(o, str) for o in components): raise TypeError("Iterable components not allowed")

        if from_type==True:
            from builtins import tuple
            components = tuple(self.type.cats[i][ix] for i,ix in enumerate(components))

        if not self.type.conforms(components): raise TypeError("Must conform to %s"%self.type.__str__())
        from builtins import tuple
        self._cats = tuple(cat.index(c) for c,cat in zip(components, self.type.cats))
        assert len(self._cats)==len(self.type.cats), "lengths do not match"

    def __str__(self):
        t = tuple(self[i] for i in range(len(self)))
        return str(t).replace("'",'')
    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, self.__str__(), len(self.type))
    def __getitem__(self, ix):
        if not isinstance(ix, int): raise TypeError("index must be an int")
        if not(-len(self._cats) <= ix < len(self._cats)): raise IndexError("out of bounds")
        value = self.type.cats[ix][self._cats[ix]]
        return value
    def __len__(self):
        return len(self._cats)

    def __eq__(self, other):
        if not isinstance(other, self.__class__): raise TypeError("Must be of class %s and not %s" % (self.__class__.__name__, other.__class__.__name__))
        if self.type != other.type: return False
        assert len(self)==len(other),"lengths do not match"
        from operator import eq, is_
        fn = is_ if (self.type.strict_eq or other.type.strict_eq) else eq
        b = all(fn(c1,c2) for c1,c2 in zip(self,other))
        return b
    def __lt__(self, other):
        if not(isinstance(other, self.__class__) and (self.type==other.type)): raise TypeError("Both must be of class {} and have the same type".format(self.__class__.__name__))
        assert len(self)==len(other),"lengths do not match"
        g = zip(self._cats, other._cats)
        from builtins import filter
        fn = lambda t : t[0]!=t[1]
        t = tuple(filter(fn, g))
        b = (not bool(t)) or (t[0][1]<t[0][0])
        return not b


def sort_categorical_tuples(list, **kwargs):
    original_list = list
    from builtins import list
    from collections.abc import Sequence

    if not all(isinstance(e, Sequence) and not isinstance(e, str) for e in original_list):
        raise TypeError("all must be tuples")
    if len(set(len(t) for t in original_list))!=1:
        class LengthDiscrepancy(Exception):pass
        raise LengthDiscrepancy("all must be of equal length")
    key = ([k for k in kwargs.keys() if ('type' in str(k).lower() or 'rank' in str(k).lower())] or ['nosuchkey'])[0]
    tp = kwargs.get(key, None)
    if tp is None:
        tp = [sorted(set(cat)) for cat in zip(*original_list)]
    tp = tp if isinstance(tp, Type) else Type(tp)

    categorical_tuple_list = [Tuple(t, type=tp) for t in original_list]
    d = {id(t):i for i,t in enumerate(categorical_tuple_list)}
    sorted_categorical_tuple_list = sorted(categorical_tuple_list)
    nx = [d[id(t)] for t in  sorted_categorical_tuple_list]
    temp_list = [original_list[ix] for ix in nx]
    original_list[:] = temp_list[:]
    return original_list


#=================================================================================

def main():
    pass
    tp = Type((1,2,3),('a','b'),(10,20,999),(1,float,int,list,str), eq_strict=True)
    tp2 = Type((1,2,3),('a','b'),(10,20,int(999.0)),(1,float,int,list,str))
    tp3 = Type((1,2,3),('a','b'),[10,20,999])
    print(tp)


    b = tp.conforms([2,'a',10,int])
    print("conforms?:", b)


    b = tp == tp2
    print(b)

    t = Tuple([3,'a',10,list], type=tp)
    t2 = Tuple(3,'a',int(999.0),list, type=tp)

    b = t==t2
    print("these Tuples are equal:", b)

    b = t<t2
    print(b)

    #TEST
    from calendar import month_abbr, day_abbr
    tp = Type(tuple(range(97,100))+tuple(['00','01','02','03']),
              month_abbr[1:],
              day_abbr[:])
    print(repr(tp))


    t1 = Tuple(99, 'Feb', 'Mon', type=tp)
    t2 = Tuple(0,1,2, type=tp, from_type=True)

    from random import randint
    l = [Tuple([randint(0,n) for n in (len(cat)-1 for cat in tp)], type=tp, from_type=True)
        for _ in range(10)]
    l = sorted(l)
    mn,mx = min(l),max(l)
    print(l)
    print(mn,mx)
    print(l[0]==l[0], mn==mx)


    tp = [(10,20,30),(True,False),(None,str,list,int), day_abbr[:]]
    t = Tuple(10,True,str,'Mon', type=tp)
    t = Tuple(0,1,2,3, from_type=True, type=tp)
    print(t, t.type)




    tp = [[1, 2, 3, 10], [2, 3, 20], [5.0, 30]]

    l = [(3,2,5.0),(3,2,5),(10,3,5), (2,20,30)]
    print(id(l), l)
    l = sort_categorical_tuples(l)
    print(id(l), l)

if __name__=='__main__': main()





