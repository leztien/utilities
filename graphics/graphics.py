from numpy import array
from matplotlib.pyplot import gca

def create_cartesian_plane(n=10, ax=None, grid=False):
    assert n>=10, "values less than 10 not supported"

    sp = ax or gca()
    sp.axis([-n,n,-n,n])

    d = dict(color='gray', width=0.002, head_length=0.3, head_width=0.3, length_includes_head = True)
    sp.arrow(-n,0,n*2,0, **d)
    sp.arrow(0,-n,0,n*2, **d)
    sp.set_aspect('equal')

    r = range(-n, n, n//10)
    sp.plot([r,r],[-n/100,n/100], color='gray', linewidth=0.6)
    sp.plot([-n/100,n/100],[r,r], color='gray', linewidth=0.6)
    [sp.text(i,-n//10*0.7, str(i), fontsize=6, ha='center') for i in range(-n,n,n//5) if i!=0]
    [sp.text(n//10*0.3, i, str(i), fontsize=6, va='center') for i in range(-n,n,n//5) if i!=0]
    if grid:
        sp.grid(True)
        sp.set_xticklabels(())
        sp.set_yticklabels(())
        sp.tick_params(axis='both', color='none')
        [sp.spines[s].set_color('gray') for s in "right,left,top,bottom".split(",")]
    else:
        sp.axis('off')
        sp.set_xticks(())
        sp.set_yticks([])
    return sp

col=0
def draw_figure(mx, ax=None, color=None):
    global col
    sp = ax or gca()
    if "ndarray" not in str(type(mx)): mx = array(mx)
    if mx.shape[0] != 2: mx = mx.T
    assert mx.shape[0]==2 and mx.shape[1]>2, "unable to infer a 2D figure from your matrix"
    lColor = ['r','g','b','yellow','lightblue','pink','darkgreen','gray','magenta']
    color = color or lColor[col]
    sp.plot(*mx, linestyle="-", marker="o", color=color)
    sp.plot(*mx[:,[0,-1]], linestyle="-", color=color)
    col = (col+1) if col <= len(lColor)-2 else 0
    return sp


def draw_vector(*args, sp=None, **kwargs):
    global plt
    def unravel(it):
        l = list()
        for e in it:
            if hasattr(e, "__len__") and type(e) is not str: l.extend(unravel(e))
            else: l.append(e)
        return l
    args = unravel(args)
    assert len(args) in (2,4), "bad arguments"
    if len(args)==2: args[0:0] = [0,0]
    xy, xytest = args[:2],args[2:]
    sp = sp or gca()

    if 'label' in kwargs:
        sp.plot(0, 0, label=kwargs['label'], color='k' if 'facecolor' not in kwargs else kwargs['facecolor'])
        kwargs.pop('label')

    d = kwargs or dict(arrowstyle="<-")
    if kwargs:
        xy, xytest = xytest, xy


    sp.annotate('', xy, xytest, arrowprops=d)
    return sp




def draw_arrow(*args, ax=None, **kwargs):
    sp = ax or gca()
    x_origin,y_origin = 0,0
    if len(args)==1 and ('sympy.matrices' in str(type(args[0])) or len(args[0])==2):
        x,y = array(args[0]).astype('f').ravel()
    elif len(args)==2:
        x,y = args
    elif len(args)==4:
        x_origin,y_origin,x,y = args
    else: raise Exception("bad arguments")
    col = kwargs.get('color', 'k')
    if 'color' in kwargs: kwargs.pop('color')
    t = ("text","name",'label','vecname','veclabel','vectorname','vec_name','vec_label','vec_text')
    l = [s for s in t if kwargs.get(s, None) is not None ]
    if len(l)>0:
        s = kwargs[l[0]]
        if len(s)==1: s = r"$\vec{%s}$" % s
        sp.text((x-x_origin)/2 - 0.2, (y-y_origin)/2 + 0.2, s, fontsize='medium')
    [kwargs.pop(s) for s in t if s in kwargs]
    sp.arrow(x_origin,y_origin,x,y, linewidth=0.5, head_length=0.18, head_width = 0.18,
          color=col, length_includes_head = True, **kwargs)
    return sp



def main():
    """TEST"""

    sp = create_cartesian_plane(10, grid=True)
    sp.set_title("title")

    l = [(1,7),(1,2),(6,2)]
    mx = array(l, dtype='i')
    print(mx)
    draw_figure(mx)
    draw_figure(mx+2)  # translation transformation

    draw_vector(1, 2, sp=sp, label="vector1")
    draw_vector(-1, -2, -3, -4.5, headwidth=10, headlength=15, facecolor="green", width=3, label="vector2")
    draw_arrow(1, 2, 3, 4, label="arrow")

    from matplotlib.pyplot import show, legend
    legend()
    show()
if __name__ == '__main__':main()