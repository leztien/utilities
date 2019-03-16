

class KMeansClassifier:
    def __init__(self, n_clusters=2):
        assert isinstance(n_clusters, int), "n_clusters must be an int"
        self.X = self.centroids_ = self._y_train_pred = None
        self.n_clusters = n_clusters


    def fit(self, X):
        from numpy import sort, unique, allclose, multiply
        from numpy.random import uniform
        from pandas import DataFrame
        from collections import deque
        Centroids_history = deque(maxlen=2)
        self.X = X
        self.centroids_ = None


        """THE TRIALS LOOP - TO SELECT THE BEST RESULT"""
        trials =[]
        for trial_number in range(5):   # max number of trials

            """GENERATE SATRTING RANDOM POINTS"""   # n_clusters points; n_dim columns
            space = sort(X, axis=0)[[0,-1],:]
            while True:
                C = current_centroids = uniform(low=space[0], high=space[1], size=(self.n_clusters, X.shape[-1]))
                y_current = ((X[None, :, :] - C[:, None, :]) ** 2).sum(axis=-1).T.argmin(axis=1)
                if len(unique(y_current)) == self.n_clusters:
                    break

            """THE INNER LOOP"""
            df = DataFrame(X, dtype=X.dtype)
            for loop_number in range(100):

                """RECALCULATE CENTROIDS"""
                dfGrouped = df.groupby(by=y_current).mean()

                """IN CASE A CENTROID IS LOST"""
                if len(unique(y_current))!=self.n_clusters:
                    lost_centroids = sorted(set(range(self.n_clusters)).difference(unique(y_current)))
                    appendage = DataFrame(C[lost_centroids], index=lost_centroids, dtype=C.dtype)
                    dfGrouped = dfGrouped.append(appendage).sort_index()
                C = dfGrouped.values

                """CHECK IF IT IS TIME TO BREAK OUT OF THE LOOP"""
                Centroids_history.append(C)
                if len(Centroids_history)>=2 and allclose(Centroids_history[-2], Centroids_history[-1]):
                    break
                else:
                    y_current = ((X[None, :, :] - C[:, None, :]) ** 2).sum(axis=-1).T.argmin(axis=1)
                #continue the loop
            else:
                from warnings import warn
                warn("max loops reached: {}".format(loop_number))

            """after (successful) breaking out of the loop: CALCULATE VARIANCE"""
            cluster_variances = df.groupby(by=y_current).var(ddof=0).sum(axis=0) ** 0.5
            hypervolume_as_metric_for_total_variance = multiply.reduce(cluster_variances.values)
            trials.append((C, y_current, hypervolume_as_metric_for_total_variance))
        else: "END OF THE TRIALS LOOP"

        """SELECTING THE BEST TRIAL"""
        from operator import itemgetter
        trials = sorted(trials, key=itemgetter(-1))
        self.centroids_ = trials[0][0]
        self._y_train_pred = trials[0][1]
        return self


    def fit_predict(self, X):
        self.fit(X)
        y_pred = self._y_train_pred
        return y_pred


    def predict(self, X):
        if self.centroids_ is None:
            raise Exception("You must fit the model first")
        y_pred = ((X[None, :, :] - self.centroids_[:, None, :]) ** 2).sum(axis=-1).T.argmin(axis=1)
        return y_pred


    @property
    def centroids(self):
        if self.centroids_ is None: raise Exception("You must fit the model first")
        from pandas import DataFrame
        INDEX = ['Centroid '+str(i+1) for i in range(self.centroids_.shape[0])]
        COLUMNS = ['Dimension '+str(i+1) for i in range(self.centroids_.shape[1])]
        df = DataFrame(self.centroids_, index=INDEX, columns=COLUMNS).round(2)
        return df



#########################################################################
#########################################################################
#########################################################################

def main():  #TEST
    import numpy as np, matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from myutils.datasets import make_multidim_blobs


    X, y = make_multidim_blobs(n_blobs=5, n_points=200, n_dim=3, range=100, relative_dispersion=10)
    n_clusters = len(np.unique(y))
    MD = KMeansClassifier(n_clusters=n_clusters)
    y_pred = MD.fit_predict(X)


    sp = plt.subplot(projection='3d')
    sp.scatter(*X.T, c=y_pred)
    plt.show()


    y_pred_again = MD.predict(X)
    b = np.all(y_pred == y_pred_again)
    print(b)

    #TEST MULTIDIM
    X, y = make_multidim_blobs(n_blobs=6, n_points=200, n_dim=5, range=1, relative_dispersion=10)
    y_true = y
    n_clusters = len(np.unique(y))
    MD = KMeansClassifier(n_clusters=n_clusters)
    y_pred = MD.fit_predict(X)
    print(y_pred)


    from sklearn.metrics import adjusted_mutual_info_score
    score = adjusted_mutual_info_score(y_true, y_pred)
    print("score:", score)


    #COMPARE WITH SKLEARN
    from sklearn.datasets import load_digits
    import sklearn.cluster
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    d = load_digits()
    X = d.data  # [:1000]
    y = y_true = d.target  # [:1000]

    print(X.shape)

    MD = sklearn.cluster.KMeans(n_clusters=10)
    y_pred = MD.fit_predict(X)
    print(len(set(y_pred)))

    score = adjusted_mutual_info_score(y_true, y_pred)
    print("score:", score)
    score = adjusted_rand_score(y_true, y_pred)
    print("score:", score)

    y_pred = KMeansClassifier(n_clusters=10).fit_predict(X)
    print(len(set(y_pred)))

    score = adjusted_mutual_info_score(y_true, y_pred)
    print("score:", score)
    score = adjusted_rand_score(y_true, y_pred)
    print("score:", score)


if __name__=='__main__':main()





