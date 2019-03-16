

def overlapping_histograms(*args, **kwargs):    # overlapping semi-transparent histograms
    from matplotlib.pyplot import gca, Axes

    #the first positional arg may be a subplot or the array
    if isinstance(args[0], Axes): sp,args = args[0], args[1:]
    else: sp = gca()

    labels = kwargs.get('labels',None) or ['data '+str(n+1) for n in range(len(args))]

    #remove key 'labels' from kwargs
    if 'labels' in kwargs: del kwargs['labels']

    #make sure the lengths match
    assert len(labels)==len(args), "number of labels doesnt match"

    #handle colours list
    from matplotlib.cm import tab10
    colors = kwargs.get('colors', None) or kwargs.get('colours', None) or [tab10(n) for n in range(len(args))]
    assert len(colors)==len(args),"number of colors does not match"

    #remove key 'color' & 'colours' from kwargs
    kwargs.pop('color') if kwargs.__contains__('color') else None
    kwargs.pop('colors') if kwargs.__contains__('colors') else None

    #let the rest of the keywords provided by the user be added to the dict which will be passes into plt.hist function
    d = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
    d.update(kwargs)

    #draw the histograms
    [sp.hist(arr, **d, label=labels[i], color=colors[i]) for i,arr in enumerate(args)]
    sp.legend()
    return sp


def main():
    import numpy as np, matplotlib.pyplot as plt
    x1 = np.random.normal(0, 0.8, 1000)
    x2 = np.random.normal(-2, 1, 1000)
    x3 = np.random.normal(3, 2, 1000)

    sp = plt.subplot()
    sp = overlapping_histograms(x1 ,x2 ,x3, bins=10)
    plt.show()

if __name__=='__main__':main()
















