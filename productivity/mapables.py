


class DictionaryForUnhashables:
    """This dictionary allows unhashable keys.
    Intended for a function caching other functions results
    (because a function's arguments/keywords can be unhashable)
    Completion date: November 2018"""
    def __init__(self, *args, copy=False, **kwargs):
        assert all(type(arg) in (list, tuple) and len(arg)==2 for arg in args), "bad argument(s)"
        self._keys_list = []     # list of tuples
        self._values_list = []
        self._copy = copy        # True = make a deepcopy of the key-tuple, so that if a mutable object in that tuple is ever changed, you will have a key discrepancy
        for (k,v) in args:
            self[k] = v
        for (k,v) in kwargs.items():
            self.__setitem__(k,v)

    def _normalize_key(self, k):
        if type(k) not in (list, tuple, set, dict): k = (k,)
        k = tuple(k)
        return k

    def __setitem__(self, k, v):
        k = self._normalize_key(k)
        if self._copy:
            from copy import deepcopy
            k = deepcopy(k)
        if k in self: del self[k]
        self._keys_list.append(k)
        self._values_list.append(v)

    def __getitem__(self, k):
        k = self._normalize_key(k)
        b = self._keys_list.__contains__(k)
        if b:
            ix = self._keys_list.index(k)
            v = self._values_list[ix]
            return v
        else:
            raise self.NoSuchEntry(what=k)

    def __delitem__(self, k):
        if not self.__contains__(k):
            raise self.NoSuchEntry(what=k)
        else:
            k = self._normalize_key(k)
            ix = self._keys_list.index(k)
            self._keys_list.pop(ix)
            self._values_list.pop(ix)

    def __contains__(self, k):
        k = self._normalize_key(k)
        b = self._keys_list.__contains__(k)
        return b

    def get(self, k, default=None):
        return self[k] if k in self else default

    def setdefault(self, k, v):
        if k not in self: self[k] = v
        return self[k]

    def get_key_by_value(self, v):
        b = self._values_list.__contains__(v)
        if b:
            ix = self._values_list.index(v)
            k = self._keys_list[ix]
            return k
        else:
            raise self.NoSuchEntry(what=v)

    def __call__(self, v):    # get key by value
        return self.get_key_by_value(v)

    def __str__(self):
        s = str.join(",  ", ("{}: {}".format(k if len(k)>1 else k[0],v) for k,v in self.items()))
        return "{%s}" %s

    def keys(self):
        return self.__iter__()

    def values(self):
        for e in self._values_list:
            yield e

    def items(self):
        for t in zip(self._keys_list, self._values_list):
            yield t

    def __iter__(self):
        for e in self._keys_list:
            yield e

    def __len__(self):
        assert len(self._keys_list)==len(self._values_list), "unexpected error: orphaned key/value"
        return len(self._keys_list)

    def __invert__(self):   # swap k:v
        return self.__class__(*((v,k) for k,v in self.items()))

    class NoSuchEntry(LookupError):
            def __init__(self, what="Value or Key", message="not found!"):
                super().__init__(str(what) + ' ' + message)
##############################################################################################################################



class TwoWayDictionary():
    def __init__(self, it=None, **kwargs):
        self._dictionary = kwargs if it is None else it if type(it) is dict else dict(it)
    
    def __str__(self):
        return str(self._dictionary) +" & " +str({i:k for k,i in self._dictionary.items()})
    
    def __getitem__(self, arg):
        if arg in self._dictionary: return self._dictionary[arg]
        elif arg in self._dictionary.values(): return [k for k,i in self._dictionary.items() if arg==i]
        else: raise KeyError("No such value in this two way dictionary!")
    
    def __setitem__(self, k, i):
        
        if i in self._dictionary.keys(): del self._dictionary[i]
        if k in self._dictionary.values(): del self._dictionary[[key for key,val in self._dictionary.items() if k==val][0]]
        self._dictionary[k] = i
        
    def __delitem__(self, arg):
        if arg in self._dictionary: del self._dictionary[arg]
        elif arg in self._dictionary.values(): del self._dictionary[[k for k,i in self._dictionary.items() if arg==i][0]]
        else: raise KeyError("No such value in this two way dictionary!")
    def remove(self, arg): self.__delitem__(arg)
    def delete(self, arg): self.__delitem__(arg)
    def pop(self, arg): self.__delitem__(arg)

    def __len__(self): return len(self._dictionary)
    
    def __contains__(self,arg):
        return (arg in self._dictionary) or (arg in {i:k for k,i in self._dictionary.items()})




class TrulyOrderedDict():
    """must include docs! e.g. date of completion etc."""
    def __new__(cls, Data=None, OrderByValue=False, **kwargs):
        from collections import OrderedDict
        if Data is None:
            try: l_t = sorted([(k,i) for k,i in kwargs.items()], key=lambda a: a[OrderByValue])
            except TypeError: l_t = [(k,i) for _,k,i in sorted([(str(i), k,i) for k,i in kwargs.items()])]
        return OrderedDict(l_t)









##############################################################################





def main():
    
    dd = TwoWayDictionary(key1=10, key2 =22)
    print(dd)
    
    i = dd['key1']      ;print(i)
    i = dd[10]          ;print(i)
    #i = dd[999]
    
    dd['key3'] = 33
    print(dd)
    
    del dd['key1']
    print(dd)
    
    del dd[22]
    print(dd)
    
    
    #---------------
    
    dd = TwoWayDictionary({1:'one', 2:'two'})
    
    print(dd)
    
    dd.remove(1)
    print(dd)
    
    dd[1] = 'one'
    print(dd)
    
    dd[1] = 'eins'
    print(dd)
    
    dd['eins'] = 2
    print(dd)
    
    dd[2] = 'zwei'
    print(dd)
    
    dd['zwei'] = 2
    print(dd)
    
    dd[2] = 'dua'
    print(dd)

    d = DictionaryForUnhashables(([1, 2], ['result', 2]), [(1, 2.5), 3], ((1, 2, 3), 1), ((2, "arg", [21, 21], {1, 2, 2}, {'a':2, 'b': 'bee', 3:3}), 2), kw1=10, copy=False)
    print(d)
    d['kw2'] = 20
    d[20] = 20
    d[[1,2,3]] = 'list'
    d[(1,2,[1,2,3])] = 300

    l = [24,434]
    d[1,2, "hjhj", [1,2,3], l] = 888

    print(d['kw2'])
    print(d[20])
    print(d[[1,2,3]])
    print(d[[1,2,[1,2,3]]])
    #d[111]



    t = (1, [1,2,3])
    v = 999
    d[t] = v

    t[1][-1] *= 10

    b = t in d
    print(b)

    print(d.get_key_by_value(999))
    print(d(999))


    d[(1,2, "keyword", [1,2,3], (3,4,6))] = 777



    print(d._keys_list)
    print(d._values_list)
    print(d[t])
    del d[t]
    #print(d[t])

    print(d._keys_list)
    print(d._values_list)


    print("===")
    for e in d:
        print(e)
    else: print("done")

    g = iter(d)
    print(*g)

    for e in d.items():
        print(e)

    print(d)
    #del d[(1,555,5)]   #error

    print(d.get(t, "NONE"))


    d[(1,{1,2,2},{'a':1,'b':2},[1,"two"])] = 555
    print(d)

    d[t] = 1111
    print(d[t])
    d[t] = 2222
    print(d[t])
    print(d)
    print(len(d))

    print(d['kw1'])
    d['kw1'] += 100
    print(d['kw1'])

    d[(1, 2)] = "test1"
    print(d)
    d[(1.0, 2)] = "test2"
    print(d)

    v = d.setdefault([10,20], "10!20")
    print(v)
    v = d.setdefault([10,20], "NEW")
    print(v)


    d = ~d
    print(d)
    print(d[3])
    print(20 in d)
    print(*d.values())


if __name__=="__main__":main()