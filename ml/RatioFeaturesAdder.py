
__version__ = '1.0'



from sklearn.base import BaseEstimator, TransformerMixin
class Subsetter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=slice(None), if_df_return_values=False):
        self.columns = columns
        self._if_df_return_values = if_df_return_values
    def fit(self, X, *args, **kwargs):
        return self
    def transform(self, X, *args, **kwargs):
        from pandas import DataFrame
        from numpy import ndarray, matrix
        if isinstance(X, DataFrame):
            df = X
            return df[self.columns].values if self._if_df_return_values else df[self.columns]
        elif isinstance(X, ndarray) or isinstance(X, matrix):
            X = X[:,self.columns]
            return X
        else: raise TypeError("ndarray, matrix or DataFrame allowed only")
        return None



from sklearn.base import BaseEstimator, TransformerMixin
class RatioFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, *args):  # when usind in a pipeline, sklearn might not like *args  >>> change to args
        def unravel(it):
            l = list()
            for e in it:
                if hasattr(e, "__len__") and type(e) is not str: l.extend(unravel(e))
                else: l.append(e)
            return l
        a = self.__unravelled_array = unravel(args)
        assert len(a)%2==0, "error1"
        assert all(isinstance(e,int) for e in a) or all(isinstance(e, str) for e in a), "error2"
        self.feature_pairs_for_ratios = [(a[i],a[i+1]) for i in range(0, len(a)-1, 2)]

    def fit(self, X, y=None):
        a = self.__unravelled_array
        if isinstance(a[0], int):
            from numpy import ndarray, matrix
            assert type(X) in (ndarray,matrix),"error3"
            assert min(a)>=0 and max(a)<X.shape[-1],"error4"
        elif isinstance(a[0], str):
            from pandas import DataFrame
            df = X
            assert isinstance(df, DataFrame),"error5"
            assert all(s in df.columns.values for s in a),"error6"
        else: raise TypeError("unforseen error")
        return self

    def transform(self, X, y=None):
        from numpy import ndarray, hstack, matrix
        from pandas import DataFrame, concat

        new_features = []
        if isinstance(X, ndarray):   #np.matrix is not yet implemented
            for ix1,ix2 in self.feature_pairs_for_ratios:
                new_feature = X[:,ix1]/X[:,ix2]  #TODO: Devision by Zero
                new_features.append(new_feature[:,None])
            X = hstack([X, *new_features])

        elif isinstance(X,DataFrame):
            df = X
            for ix1,ix2 in self.feature_pairs_for_ratios:
                new_feature = df[ix1] / df[ix2]   #TODO: Devision by Zero
                new_feature.name = df[ix1].name + ' / ' + df[ix2].name
                new_features.append(new_feature)
            df = X = concat([df,*new_features], axis=1)
        else: raise TypeError("your type is not implemented")
        return X




def main():
    import numpy as np, pandas as pd

    X = np.arange(10*5).reshape(5,10).T
    df = pd.DataFrame(X, columns=list("ABCDE"))

    tr = RatioFeaturesAdder(0,1,1,2)
    Xtr = tr.fit_transform(X).round(1)
    print(Xtr)

    tr = RatioFeaturesAdder(["A","B","C","D","A","B"])
    tr.fit(df)
    df = tr.fit_transform(df)
    print(df)

if __name__=='__main__':main()




