




class Coord:
    def __init__(self, hours, minutes=None):
        minutes = minutes or 0
        assert -180 <= hours <= 180, "bar hours provided"
        assert 0 <= minutes < 60, "bad mintes provided"
        self.hours, self.minutes = hours, minutes
    
    def __str__(self):
        s = "{:>02}{}{:>02}{}".format(self.hours, chr(176), self.minutes, chr(8242))
        return s
    def __repr__(self):
        return self.__str__()
    
    def __float__(self):
        minutes = self.minutes / 60
        f = (abs(self.hours) + minutes) * (-1 if self.hours<0 else 1)
        return f
    
    def __add__(self, other):  
        if isinstance(other,int):
            totalminutes = (abs(self.hours)*60 + self.minutes) * (-1 if self.hours<0 else 1)
            totalminutes += other
            h,m = divmod(abs(totalminutes), 60)
            h *= -1 if totalminutes<0 else 1
            ins = self.__class__(h,m)
            return ins
        elif isinstance(other,self.__class__):
            return NotImplemented
        else: raise NotImplementedError
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return self.__add__(-other)
    
    def __iadd__(self, other):  
        if isinstance(other,int):
            totalminutes = (abs(self.hours)*60 + self.minutes) * (-1 if self.hours<0 else 1)
            totalminutes += other
            h,m = divmod(abs(totalminutes), 60)
            h *= -1 if totalminutes<0 else 1
            self.hours, self.minutes = h,m
            return self
        elif isinstance(other,self.__class__):
            return NotImplemented
        else: raise NotImplementedError
    
    def __isub__(self, other):
        return self.__iadd__(-other)
        
    @classmethod
    def fromstring(cls, string):
        from re import compile
        pt = compile(r"(-?\d{1,2})%s(\d{1,2})" % chr(176))
        m = pt.match(string.strip())
        if m:
            t = m.groups()
            if all(s.lstrip('-').isdecimal() for s in t):
                h,m = (int(n) for n in t)
                ins = cls(h,m)
                return(ins)
        raise ValueError("parsing failed")
#################END OF OORD CLASS###########################################################


def make_data(m=100):
    import numpy as np, pandas as pd
    rs = np.random.RandomState(0)
    
    #FEATURE x1 and x2 (latitude, longitude)
    hamburg = [Coord.fromstring(s) for s in "53°33 10°00".split()]
    latitude_longitude = rs.uniform([-20,-20], [20,20], (m,2)).astype(int).astype(object)
    nd = hamburg + latitude_longitude
    vectorized_float = np.vectorize(float)
    nd = vectorized_float(nd).astype(np.float16).round(2)  #feature x1, x2
    mean_dist = (((nd - [float(e) for e in hamburg]) **2).sum(axis=1)**0.5).mean()  #to be used as: if > dist
    is_outside_citycenter = (((nd - [float(e) for e in hamburg]) **2).sum(axis=1)**0.5) > mean_dist
    x1,x2 = nd.T
    f12 = np.abs(rs.normal(loc=nd.mean(), scale=nd.std(), size=m))
    
    #FEATURE x3 (site side length in meters)
    x3 = rs.normal(loc=10, scale=3, size=m).round(1)
    x3[x3<=0] = 3
    f3 = np.abs(rs.normal(loc=x3.mean(), scale=x3.std(), size=m))
    
    #FEATURE x4 (total costs of the site)
    price_per_meter = rs.normal(loc=10000, scale=2000, size=m)
    x4 = (x3 * price_per_meter).round()
    f4 = np.abs(rs.normal(loc=x4.mean(), scale=x4.std(), size=m)).round()
    
    #FEATURE (rating)
    x5 = rs.choice(['A','B','C'], m, replace=True)
    
    #DATAFRAME
    pd.options.display.precision=2
    df = pd.DataFrame({'latitude':x1.round(2), 'longitude':np.round(x2,2), 
                       'side length':x3, 'site costs':x4, 'rating':x5,
                       'f12':f12, 'f3':f3, 'f4':f4}).round(2)
       
    X = np.vstack([is_outside_citycenter, x3**3, x4/x3, pd.factorize(x5)[0]]).T
    EE = X.mean(axis=0)  #Expected values
    MX = EE.max()
    weights = MX / EE
    weights *= [-1, 1, -1, 1] 
    
    y = (X * weights).sum(axis=1)
    y = X @ weights   #same
    θ = np.array([np.abs(y.min()), *weights]).round(2).astype(np.float16)
    y += np.abs(y.min())
    mask = y>0
    X,y,df = (arr[mask] for arr in (X,y,df))
    
    
    from sklearn.linear_model import LinearRegression
    md = LinearRegression().fit(X,y)
    print("score:::", md.score(X,y))
    print(X)
    print(y)
    #np.save("X", X)
    #np.save("y", y)
    print(mean_dist)
    #APPEND THE TARGET TO DATAFRAME
    df['profit'] = y
    return df

###############END OF MAKE DATA#######################################

def add_outliers(a, p=None, copy=True, random_state=None):
    from numpy.random import RandomState, randint, randn
    from numpy import quantile, abs, logical_or, setdiff1d
    if copy: a = a.copy() 
    p = p or 0.05; assert isinstance(p, float) and 0<p<1,"error1"
    
    rs = RandomState(random_state or randint(0, int(1e6)))
    
    Q1,Q3 = quantile(a, q=[0.25, 0.75])
    IQR = Q3 - Q1
    LB = Q1 - IQR*1.5
    UB = Q3 + IQR*1.5
    
    nxOriginalOutliers = logical_or(a<LB,a>UB).nonzero()[0]
    how_many = int(round(len(a)*p,0)) - len(nxOriginalOutliers)
    nx = set(range(len(a))) - set(nxOriginalOutliers)
  
    nx = rs.choice(list(nx), replace=False, size=how_many)
    nx = setdiff1d(nx, nxOriginalOutliers)
 
    add_outliers.original_outliers = nxOriginalOutliers.tolist()
    add_outliers.added_outliers = []
    
    if len(nx)==0 or how_many<1: 
        from warnings import warn
        warn("No outliers were added", Warning)
        return a
    
    nxL,nxU = nx[:how_many//2], nx[how_many//2:]
    a[nxL] = LB - abs(randn(len(a[nxL]))*(a.std()/1.5))
    a[nxU] = UB + abs(randn(len(a[nxU]))*(a.std()/1.5))
    add_outliers.added_outliers = nx.tolist()
    
    Q1,Q3 = quantile(a, q=[0.25, 0.75])
    IQR = Q3 - Q1
    LB = Q1 - IQR*1.5
    UB = Q3 + IQR*1.5
    add_outliers.actual_outliers = logical_or(a<LB,a>UB).sum()
    return a

def add_outliers_to_matrix(mx, columns=None, p=None):  #mx is passes by ref
    columns = columns or range(mx.shape[-1])
    if not hasattr(columns, '__len__'): columns = [columns,]
    p = p if hasattr(p,'__len__') else [p,]*len(columns)
    assert len(columns)==len(p),"error10"
    for j,p in zip(columns,p):
        mx[:,j] = add_outliers(mx[:,j], p, copy=True)
    return(mx)  #if nothing is return the passed in mx will still be modified
    
#========================================================================================

def makedata(m=100, n_target_categories=None, add_nans=False, add_noise=False, add_outliers=False, return_csv_filepath=True, random_seed=None):
    import numpy as np  # e.g. filepath = makedata(1000, return_csv_filepath=True, add_nans=0.01, add_noise=True, add_outliers=True)
    import scipy.stats
    from pandas import DataFrame, factorize, Categorical
    
    rs = np.random.RandomState(random_seed or 0)
    
    #FEATURE x1 (side-length)
    probs = rs.randint(5,21, size=10)
    probs = probs / probs.sum()
    x1 = rs.choice([*range(1,11)], p=probs, size=m, replace=True)
    
    
    #FEATURE x2 (thickness)
    multinomail = scipy.stats.multinomial(m, [0.1, 0.2, 0.3, 0.25, 0.15])
    counts = multinomail.rvs(1)[0].tolist()  
    l = [[n,]*k for n,k in zip(range(5,11), counts)]   #vizualize the distribution here
    l = sum(l,[])
    import random
    random.shuffle(l)
    x2 = (np.array(l) + x1**2*0.05).round(1)
    
    
    #FEATURE x3 (total >>> ratio)  in $
    a = rs.normal(loc=100, scale=25, size=m)
    x3 = (a * x1).round(2)
    
    #FEATURE x4  (yes/no = 50/50)
    x4 = rs.choice(['no','yes'], size=m, replace=True)
    
    #FEATURE x5  (garde)
    x5 = scipy.stats.binom.rvs(n=2, p=0.5, size=m)
    x5 = np.array(list("CBA"))[x5]
    
    #FAKE FEATURE x6
    x6 = x1 * x2
    x6 = rs.normal(loc=x6.mean(), scale=x6.std(), size=m).round(2)
    
    #FAKE FEATURE x7
    x7 = np.abs(rs.normal(loc=x3.mean(), scale=x3.std(), size=m)).round(2)
    
    #TARGET
    f1 = x1**2
    f2 = x1**2*x2
    f3 = x3/x1
    f4 = np.abs(factorize(x4)[0]-1).astype(np.uint8)
    f5 = Categorical(x5, ordered=True).reorder_categories(list("CBA")).rename_categories([0,1,2]).astype(np.uint8)
    X = np.vstack([f1,f2,f3,f4,f5]).T
    
    μμ = X.mean(axis=0)
    mx = μμ.max()
    θ = mx / μμ
    y = (X @ θ).round(2) + (rs.randn(m) if add_noise else 0)
    
    #ADD OUTLIERS
    if add_outliers:
        add_outliers = globals()['add_outliers']
        x2 = add_outliers(x2, p=0.03, random_state=rs.get_state()[1][0]) #assume erronious measurement
        y = add_outliers(y, p=0.03, random_state=rs.get_state()[1][0]+1) #assume erronious measurement

    #IF CATEGORICAL TARGET
    if n_target_categories:        
        assert isinstance(n_target_categories,int)and(2<=n_target_categories<=100),"bad categories, must be int"
        from numpy import linspace,digitize
        breaks = linspace(y.min(), y.max()+0.001, n_target_categories+1)
        y = digitize(y, breaks) - 1
        
    #DATAFRAME
    df = DataFrame({'length':x1,'thickness':x2,'total':x3,'yes/no':x4,'grade':x5,
                    'feature6':x6, 'feature7':x7, 'target':y})
    
    #ADD NAN'S
    if add_nans:
        p = add_nans if isinstance(add_nans, float) else 0.01
        subset = slice(None),slice(1,None,None)
        def indeces_for_random_nan_cells(df, p=0.1, subset=None):
            from numpy import dtype, uint8, zeros, random
            subset = subset or (slice(None),slice(None))
            
            m,n = df.shape
            
            dtype = dtype([('row', uint8), ('col', 'uint8')])
            nd = zeros(shape=(m,n), dtype=dtype)
            
            [nd[i].__setitem__('row', i) for i in range(m)]
            [nd[:,j].__setitem__('col', j) for j in range(n)]
            
            nd = nd[subset]
            
            k = int(round(nd.size * p))   # number of cells to fill up with nan's
            nx = random.choice(nd.size, replace=False, size=k)
            nx = nd.ravel()[nx]
            nx = [tuple(t) for t in nx]
            return(nx)
        nx = indeces_for_random_nan_cells(df, p, subset)
        [df.iloc.__setitem__(t, np.nan) for t in nx]
        
        if return_csv_filepath:
            import os, time
            DIR = os.environ.get('TEMP')
            FILENAME = "temp_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
            fullfilename = os.path.join(DIR, FILENAME)
            df.to_csv(fullfilename, index=False)
            del df
            return fullfilename
    return(df)

#=============================================================================================

def prepare_data_and_split_into_X_y(df, recursive_elimination=True, columns=None) -> ('df','y'):    # returns a df with the X-values and the target y-variable
    from utility import OutliersRemover, DropTargetNans
    from sklearn.pipeline import Pipeline
    columns = columns or [1,2,5,6,7]
    outliersremover = OutliersRemover(columns=columns, recursive_elimination=recursive_elimination)
    droptargetnans = DropTargetNans()
    pl = Pipeline([('outliersremover', outliersremover), ('droptargetnans', droptargetnans)])
    dfOutliersAndNansRemoved = pl.fit_transform(df)

    dfX = dfOutliersAndNansRemoved.iloc[:,:-1]

    y = ytrue = dfOutliersAndNansRemoved['target'].values
    return(dfX,y) 
    
#========================================================================================
    
from sklearn.base import BaseEstimator, TransformerMixin
class Subsetter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, if_df_return_values=False, astype_float=False):
        self.columns = columns or slice(None)
        self._if_df_return_values = if_df_return_values
        self.astype_float = astype_float
    def fit(self, X, *args, **kwargs):
        return self
    def transform(self, X, *args, **kwargs):
        from pandas import DataFrame
        from numpy import ndarray, matrix, float64
        
        if type(X)in(ndarray,matrix):
            X = X[:,self.columns]
        
        elif isinstance(X, DataFrame):
            df = X
            from numbers import Integral
            if hasattr(self.columns, '__iter__') and all(isinstance(n,Integral) for n in self.columns):
                df = df.iloc[:,self.columns]
            else:    
                df = df[self.columns]
            
            if not self._if_df_return_values: return df
            else: X = df.values 
        else: raise TypeError("unforseen error / not supported type")
        
        X = X.astype(float64) if self.astype_float else X
        return X
 

#======================================================================================

from sklearn.base import BaseEstimator, TransformerMixin
class RatioFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, column_pairs=None, return_ratio_only=False, if_df_return_values=False):  # when usind in a pipeline, sklearn might not like *args
        self.column_pairs = column_pairs or [(0,1),]
        self.return_ratio_only = return_ratio_only
        self.if_df_return_values = if_df_return_values

    @property
    def column_pairs(self):
        return self.__column_pairs
    @column_pairs.setter
    def column_pairs(self, new):
        assert hasattr(new, '__iter__'),"must be iterable"
        def unravel(it):
            l = list()
            for e in it:
                if hasattr(e, "__len__") and type(e) is not str: l.extend(unravel(e))
                else: l.append(e)
            return l
        
        a = unravel(new)
        assert len(a)>=2,"error1"
        assert len(a)%2==0, "error2"
        assert all(isinstance(e,int) for e in a) or all(isinstance(e, str) for e in a), "error3"
        new = [(a[i],a[i+1]) for i in range(0, len(a)-1, 2)]
        self.__column_pairs = new
        self.__unraveled_column_pairs = a   #for inner purposes
    
    
    def fit(self, X, y=None, *args, **kwargs):
        a = self.__unraveled_column_pairs
        if isinstance(a[0], int):
            from numpy import ndarray, matrix; from pandas import DataFrame
            assert type(X) in (ndarray,matrix) or (isinstance(X, DataFrame)and self.if_df_return_values),"error4"
            assert min(a)>=0 and max(a)<X.shape[-1],"error5"
        elif isinstance(a[0], str):
            from pandas import DataFrame
            df = X
            assert isinstance(df, DataFrame),"error6"
            assert all(s in df.columns.values for s in a),"error7"
        else: raise TypeError("unforseen error")
        return self

    def transform(self, X, y=None, *args, **kwargs):
        from numpy import ndarray, hstack, nan, isinf
        from pandas import DataFrame, concat
        if isinstance(X, DataFrame) and self.if_df_return_values and isinstance(self.__unraveled_column_pairs[0], int):
            X = X.values
        self.fit(X)  # in order to catch possible errors when transforming new data
        
        new_features = []
        if isinstance(X, ndarray):   #np.matrix is not yet implemented
            for ix1,ix2 in self.column_pairs:
                a1,a2 = (arr.astype('f') for arr in (X[:,ix1],X[:,ix2] ))
                new_feature = a1/a2
                new_feature[isinf(new_feature)] = nan
                new_features.append(new_feature[:,None])
            if self.return_ratio_only: return hstack([*new_features])
            else: return hstack([X, *new_features])

        elif isinstance(X,DataFrame):
            df = X
            for ix1,ix2 in self.column_pairs:
                new_feature = df[ix1] / df[ix2]   
                #TODO: Devision by Zero
                new_feature.name = df[ix1].name + ' / ' + df[ix2].name
                new_features.append(new_feature)
            if self.return_ratio_only: return concat([*new_features], axis=1)
            else: return concat([df,*new_features], axis=1)
        else: raise TypeError("your type is not implemented")
        raise Exception("unforseen error")


#=====================================================================================

from sklearn.base import BaseEstimator, TransformerMixin
class RatioFeaturesSimple(BaseEstimator, TransformerMixin):
    def __init__(self, column_pairs=None, return_ratio_only=False, if_df_return_values=False):  # when usind in a pipeline, sklearn might not like *args
        self.column_pairs = column_pairs 
        self.return_ratio_only = return_ratio_only
        self.if_df_return_values = if_df_return_values
   
    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        from numpy import ndarray, hstack, nan, isinf
        from pandas import DataFrame, concat
        if isinstance(X, DataFrame) and self.if_df_return_values and isinstance(self.__unraveled_column_pairs[0], int):
            X = X.values
        
        new_features = []
        if isinstance(X, ndarray):   #np.matrix is not yet implemented
            for ix1,ix2 in self.column_pairs:
                a1,a2 = (arr.astype('f') for arr in (X[:,ix1],X[:,ix2] ))
                new_feature = a1/a2
                new_feature[isinf(new_feature)] = nan
                new_features.append(new_feature[:,None])
            if self.return_ratio_only: return hstack([*new_features])
            else: return hstack([X, *new_features])

        elif isinstance(X,DataFrame):
            df = X
            for ix1,ix2 in self.column_pairs:
                new_feature = df[ix1] / df[ix2]   
                #TODO: Devision by Zero
                new_feature.name = df[ix1].name + ' / ' + df[ix2].name
                new_features.append(new_feature)
            if self.return_ratio_only: return concat([*new_features], axis=1)
            else: return concat([df,*new_features], axis=1)
        else: raise TypeError("your type is not implemented")

#=====================================================================================
        
from sklearn.base import BaseEstimator,TransformerMixin
class Scaler(BaseEstimator,TransformerMixin):
    """allows to scale or not to scale - for grid-search"""
    def __init__(self, scaler, scale=False):
        self.scaler = scaler # user must pass in a scaler like sklearn-MinMaxScaler()
        self.scale = scale
    
    def fit(self, X, y=None, *args):
        if self.scale:
            self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None, *args):
        if self.scale:
            X = self.scaler.transform(X)
        return X

#======================================================================================

def splitdata(df):
    sr = df['target']
    breaks = sr.quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    from numpy import digitize
    f = digitize(sr.values, breaks)

    from sklearn.model_selection import train_test_split
    dfTrain,dfTest = train_test_split(df, test_size=0.2, stratify=f)
    
    #pickle
    dfTrain.to_pickle(r"trainset_dataframe.pkl")
    dfTest.to_pickle(r"testset_dataframe.pkl")
    splitdata.train = r"trainset_dataframe.pkl"
    splitdata.test = r"testset_dataframe.pkl"
    return(dfTrain,dfTest)
    
#===================================================================================

def outlier_indeces(a, return_boolean_mask=False):
    from numpy import quantile, logical_or, nonzero
    Q1,Q3 = quantile(a, q=[0.25,0.75])
    IQR = Q3-Q1
    LB = Q1-IQR*1.5
    UB = Q3+IQR*1.5
    mask = logical_or(a<LB, a>UB)
    if return_boolean_mask: return mask
    nx = nonzero(mask)[0]
    return nx

#===================================================================================

from numpy import isnan, quantile, ndarray, floating, float64, array
from numexpr import evaluate
def outlier_bounds(a):  # get outlier bounds by recursive outlier elimination
    if not isinstance(a, ndarray): a = array(a, dtype=float64)
    if not issubclass(a.dtype.type, floating): a = a.astype(float64)

    a = a[~isnan(a)]
    Q1,Q3 = quantile(a, q=[0.25,0.75])
    IQR = Q3-Q1
    LB,UB = Q1-IQR*1.5, Q3+IQR*1.5
    
    mask = evaluate("(a<LB) | (a>UB)")  #outliers

    if int(mask.sum()): return(outlier_bounds(a[~mask]))
    return(LB,UB) #this is in effect the else-block 
    
#====================================================================================    
    
from sklearn.base import BaseEstimator, TransformerMixin
class OutliersRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, recursive_elimination=False):
        self.columns = columns or []
        self.recursive_elimination = recursive_elimination
        assert isinstance(columns,list),"must be a list"
        
    def fit(self, X, y=None, *args, **kwargs):
        from pandas import DataFrame
        from numpy import empty
        if isinstance(X, DataFrame): X = X.values
        bounds = empty(shape=(2, len(self.columns)), dtype='float64')
        for j,column in enumerate(self.columns):
            bounds[:,j] = self._outlier_bounds(X[:,column])
        self.LB, self.UB = bounds  #lower and upper bounds
        return self
    
    def transform(self, X, y=None, *args, **kwargs):
        from pandas import DataFrame
        from numpy import logical_and
        from numexpr import evaluate
        self.X = X    #keep the original input data for later
        X = X.iloc[:, self.columns].values if isinstance(X, DataFrame) else X[:,self.columns]
        LB,UB = self.LB, self.UB
        #MASK = (X <= LB)|(X >= UB); MASK = np.logical_or(X<=LB, X.__ge__(UB))  #same
        MASK = evaluate("(X<=LB)|(X>=UB)")  # X = subset     MASK = 2D mask-array denoting outliers
        self.mask = logical_and.reduce(~MASK, axis=1)  # non-outliers
        self.X = self.X.iloc[self.mask,:] if isinstance(self.X, DataFrame) else self.X[self.mask,:]
        self.number_of_outliers_removed = X.shape[0] - self.X.shape[0]
        self.outliers_index = (~self.mask).nonzero()[0]
        del X, self.mask
        return self.X
    
    from numpy import isnan, quantile, ndarray, floating, float64, array, empty
    from numexpr import evaluate
    def _outlier_bounds(self, a):  # get outlier bounds by recursive outlier elimination
        if not isinstance(a, ndarray): a = array(a, dtype=float64)
        if not issubclass(a.dtype.type, floating): a = a.astype(float64)
        mask = isnan(a)
        if mask.any(): a = a[~mask]
        Q1,Q3 = quantile(a, q=[0.25,0.75])
        IQR = Q3-Q1
        LB,UB = Q1-IQR*1.5, Q3+IQR*1.5
        
        if self.recursive_elimination:
            mask = evaluate("(a<LB) | (a>UB)")  #outliers
            if int(mask.sum()):
                return(self._outlier_bounds(a[~mask]))
        
        return(LB,UB) #this is in effect the else-block 
        
#=========================================================================================================

from sklearn.base import BaseEstimator, TransformerMixin
class DropTargetNans(BaseEstimator, TransformerMixin):  
    def __init__(self, target=None):
        self.target = target or -1
        
    def fit(self, X, y=None, *args, **kwargs):
        return self
    
    def transform(self, X, y=None, *args, **kwargs):
        from numpy import ndarray, matrix
        from pandas import DataFrame
        
        if isinstance(X, DataFrame):
            df = X 
            self.target = self.target if isinstance(self.target,str) else df.columns.values[self.target]
            df = df.dropna(subset=[self.target])
            return(df)
                
        if type(X) in (ndarray,matrix):
            from numpy import isnan
            from numbers import Integral
            assert isinstance(self.target, Integral),"must be int"
            mask = ~isnan(X[:,self.target])
            X = X[mask,:]
            return X
        else: raise TypeError("not implemented")
        
#==================================================================================================

from sklearn.base import BaseEstimator, TransformerMixin
class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, hyperparameter=None):
        self.hyperparameter = hyperparameter or [(0,1),(2,3)]
    
    @property
    def hyperparameter(self):
        return self.__hyperparameter
    @hyperparameter.setter
    def hyperparameter(self, new):
        from numbers import Integral
        b = hasattr(new, '__len__') and all(hasattr(e, '__len__')and len(e)==2 and all(isinstance(n,Integral) and n>=0 for n in e) for e in new)
        if not b: raise ValueError("hyperparameter must be of format [(0,1),(2,3), ...]")
        self.__hyperparameter = new
    
    def fit(self, X, y=None, *args, **kwargs):
        return self
    def transform(self, X, y=None, *args, **kwargs):
        return X
    
#=========================================================================================
        
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
class PolynomialExpansion(BaseEstimator, TransformerMixin):
    """returns only the polynomially expanded features, but not the original (input) features"""
    def __init__(self, degree=2, with_interactions=True):
        self.degree = degree
        self.with_interactions = with_interactions
    
    def fit(self, X, y=None, *args):
        return self
    
    def transform(self, X, y=None, *args):
        poly = PolynomialFeatures(degree=self.degree, interaction_only=not self.with_interactions, include_bias=False)
        XX = poly.fit_transform(X)
        XX = XX[:, X.shape[1]:]
        self.names = poly.get_feature_names()[X.shape[1]:]
        del poly
        return XX

#====================================================================================

from sklearn.base import BaseEstimator,TransformerMixin
class CorrelationFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self, k_best=None, proportion=None):
        self.k_best, self.proportion = k_best, proportion
        self.nx = None
    
    def fit(self, X, y, *args):
        n = self.k_best or int(round(self.proportion*X.shape[1]))
        assert isinstance(n, int)and n<X.shape[1],"error1"
        from numpy import corrcoef, c_
        r = corrcoef(c_[X,y].T).round(3)[:-1,-1]  #r = correlation coefficients
        self.nx = r.argsort()[:-(n+1):-1]
        return self
    
    def transform(self, X, y=None, *args):
        if self.nx is None: raise Exception("you must fit the model first")
        return X[:,self.nx]

#=====================================================================================

def discretize(a, bins=3):   # bins = number of equal bins
    from math import ceil    # equivelent to numpy.digitize(a, bins=breaks)
    mn,mx = min(a), max(a)   # equivelent to pandas.cut(a, bins=3) 
    nx = [ceil((n-mn)/(mx-mn)*bins) for n in a]
    nx[0] = nx[1]
    return nx  #Note discretization depends on the range of the array

#====================================================================================
    




    






