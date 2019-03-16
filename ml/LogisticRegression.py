

def logistic_regression_by_gradient_descent(X,y):
    from numpy import zeros, exp, hstack, ones, allclose
    m = X.shape[0]
    θ = zeros(shape=X.shape[-1]+1)
    X = hstack([ones(m)[:,None],X])
    α = 0.05  #learning rate
    e = exp(1)
    k = 5      #how many pockets in the deque aka stack-array should be
    stack = __import__('collections').deque(maxlen=k)

    for loop in range(int(3e+5)):
        theta_old = θ.copy()
        θ -= (α/m) * ( X.T @ (1/(1+e**-(X@θ)) - y))

        b = allclose(θ, theta_old, rtol=1e-1)
        stack.append(b)
        if len(stack)>=k and set(stack)=={True}:break

    this = globals()[__import__('inspect').stack()[0].function]
    this.loop = this.loops = loop   #number of loops
    this.probabilities = 1/(1+e**-(X @ θ))
    this.ypred = (this.probabilities >= 0.5).astype(int)
    this.probabilities = this.probabilities.round(4).tolist()
    this.score = (y==this.ypred).sum() / len(y)
    this.weights = θ.tolist()
    this.predict = lambda X : (1/(1+e**-(hstack([ones(X.shape[0])[:,None],X]) @ this.weights)) >= 0.5).astype(int)
    return(θ)



def main():
    import numpy as np
    np.set_printoptions(suppress=True, precision=4)

    m,n = 30,3
    X = np.random.randint(0, 100, size=(m,n)).astype(float)
    X[:,0]+=10
    X[:,1]*=3
    y = ytrue = (X.sum(axis=1) >= X.sum(axis=1).mean()).astype(np.uint8)

    from sklearn.linear_model import LogisticRegression
    md = LogisticRegression(solver='liblinear', C=1.0).fit(X,y)
    score = md.score(X,y)
    weights = md.intercept_, md.coef_
    ypred = md.predict(X)
    print(score, weights)
    print(ypred)



    weights = logistic_regression_by_gradient_descent(X,y)
    score = logistic_regression_by_gradient_descent.score
    print(score, weights)

    ypred = logistic_regression_by_gradient_descent.ypred
    print(ypred)

    probs = logistic_regression_by_gradient_descent.probabilities
    print("probabilities:", probs)

    loop = logistic_regression_by_gradient_descent.loop
    print("loop#", loop)

if __name__=='__main__':main()
