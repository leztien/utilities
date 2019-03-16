
from sympy import symbols, Matrix, eye, init_printing; init_printing(use_unicode=True)

def visualize_vector_transformation(T, v, ax=None, show_plot=True):
    from numpy import matrix, hstack
    from matplotlib.pyplot import show, gca, gcf
    sp = ax or gca()
    d = dict(angles='xy', scale_units='xy', scale=1, width=0.003, headwidth=3, headlength=4, alpha=0.7)

    def draw_vector(*args, **kwargs):
        color = kwargs.get('color', 'black')
        width = kwargs.get('width', 0.003)
        if len(args) == 2:
            x, y = args
        else:
            raise Exception("error")
        sp.quiver(0, 0, x, y, color=color, width=width, angles='xy', scale_units='xy', scale=1, headwidth=3,
                  headlength=4, alpha=0.7)

    x_unit = matrix([[1], [0]])
    y_unit = matrix([[0], [1]])
    draw_vector(1, 0, color='green', width=0.002)
    draw_vector(0, 1, color='red', width=0.002)
    draw_vector(*sum((matrix(T).astype('f') * x_unit).tolist(), []), color='green', width=0.004)
    draw_vector(*sum((matrix(T).astype('f') * y_unit).tolist(), []), color='red', width=0.004)
    draw_vector(*sum(matrix(v).astype('f').tolist(), []), color='lightblue', width=0.008)
    draw_vector(*sum((matrix(T).astype('f') * matrix(v).astype('f')).tolist(), []), color='blue', width=0.006)

    n = max(abs(float(e)) for e in sum(hstack([T, v, matrix(T).astype('f') * matrix(v).astype('f')]).tolist(), [])) + 1
    sp.axis([-n, n, -n, n])
    sp.axhline(0, color='gray', alpha=0.9, linewidth=0.5)
    sp.axvline(0, color='gray', alpha=0.9, linewidth=0.5)

    sp.set_aspect("equal")
    if show_plot:
        gcf().set_figheight(8)
        gcf().set_figwidth(8)
        show()
    return sp
#===============================


def main():
    a,b,c,d = symbols("a,b,c,d")
    x,y =symbols("x y")
    v = Matrix([x,y])
    T = Matrix([[a,b],
               [c,d]])
    I = eye(2)
    w = T*v
    e = I*v


    T = Matrix([[-2, -.5],
                [-1, -1]])
    v = Matrix([1,1])

    visualize_vector_transformation(T, v)

if __name__=='__main__':main()