def qqplot(a, sp=None):
    from matplotlib.pyplot import plot, gca, sca
    from scipy.stats import norm
    from statistics import mean, stdev

    μ ,σ = mean(a), stdev(a)
    a = [( v -μ ) /σ for v in sorted(a)]

    n = len(a)
    p = 1/ (n + 1)
    pp = [p * n for n in range(1, n + 1)]
    zz = norm.ppf(pp, loc=0, scale=1)

    sca(sp or gca())
    plot(a, zz, 'k.')
    gca().set(title="qqplot", xlabel="provided values normalized", ylabel="qq values")
    return (gca(), a, zz)