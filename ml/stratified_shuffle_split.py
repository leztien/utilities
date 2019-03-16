def stratified_shuffle_split(X, split_on=['target array (if clasification problem)' or 'categorical feature'], test_size=0.5):
    assert all(not isinstance(e, float) for e in split_on), "error1"
    assert len(X)==len(split_on), "error2"
    assert isinstance(test_size, float) and 0 < test_size < 1, "error3"
    
    y = split_on   # y here is the categorical array
    nx = tuple(range(len(X)))
    cats = sorted(set(y))
    nxs = [[] for _ in range(len(cats))]
    
    #put the indeses into its own category
    for ix,y in zip(nx,y):
        nxs[cats.index(y)].append(ix)
    
    #shuffle the indeces in each category
    from random import shuffle
    [shuffle(l) for l in nxs]
    
    #create nx_train, nx_test; handle the test_size/train_size percentage
    nx_train, nx_test = [],[]
    p = 1 - test_size
    
    #fill the train and test lists with indeces from each category
    for nx in nxs:
        n = round(len(nx)*p)
        nx_train.extend(nx[:n])
        nx_test.extend(nx[n:])
    
    #sort the train and test indeses for aesthetics
    nx_train, nx_test = (sorted(nx) for nx in (nx_train, nx_test))
    return nx_train, nx_test

#=============================================================================

def main():
    import numpy as np
    
    x1 = [10,20,30,40,50,60,70,80,90,99]
    x2 = [1,2,3,1,2,3,1,2,3,3]
    y = [0,1,0,1,0,1,0,1,0,1]
    
    X = np.array([x1,x2], dtype=np.uint8).T
    y = np.array(y, dtype=np.uint8)
    
    nx_train, nx_test = stratified_shuffle_split(X, split_on=x2, test_size=0.4)
    print("train set indeces:", nx_train, " test set indeces:", nx_test)
    
    print("the split-on values in the train set", X[nx_train,1])
    print("the split-on values in the test set ", X[nx_test,1])
    
    #another test
    from pandas import DataFrame
    m = 100
    df = DataFrame({'x1':tuple(range(m)),
                    'x2':np.random.choice(['a','b','c'], size=m, replace=True),
                    'y':np.random.randint(0,5, size=m)})
    
    nx_train, nx_test = stratified_shuffle_split(df, split_on=df['y'].values, test_size=0.25)
    
    sr_train = df.iloc[nx_train,-1]
    sr_train = sr_train.value_counts() / len(sr_train)
    
    sr_test = df.iloc[nx_test,-1]
    sr_test = sr_test.value_counts() / len(sr_test)
    
    sr_whole = df.iloc[:,-1]
    sr_whole = sr_whole.value_counts() / len(df)
    
    df = DataFrame([sr_whole, sr_train, sr_test], index=['whole','train','test']).round(2).T
    df.columns.name = 'sets'
    df.index.name = 'categorical values'
    print(df)
    
    print("train/test sets size ratio:", len(nx_train)/m, len(nx_test)/m)
if __name__=='__main__':main()