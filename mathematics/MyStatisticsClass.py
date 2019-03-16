import statistics, math, numpy, scipy.stats
import numpy, pandas
from re import sub
import inspect
from functools import partial
pandas.options.display.width = 170

PLOT = 'plot'   #for enum purposes
BOTH = 'both'; LOWER='lower'; UPPER='upper'
DOUBLERECIPROCALMODEL, DOUBLESQUAREROOTMODEL, EXPONENTIALMODEL, LINEARMODEL, LOGARITHMICMODEL, POWERMODEL, QUADRATICMODEL, RECIPROCALMODEL, SQUAREROOTMODEL = ['DoubleReciprocalModel', 'DoubleSquareRootModel', 'ExponentialModel', 'LinearModel', 'LogarithmicModel', 'PowerModel', 'QuadraticModel', 'ReciprocalModel', 'SquareRootModel']

class RegressionModels():
    """
    This code the most appropriate regression model for the X and Y arrays you provide.
    obj = RegressionModel(X, Y)    #both X and Y must be arrays of equal length containing only numbers
    sr = obj.FindBestModel()
    obj.Plots()                    #shows regression plots with all possible models (covered in this class)
    
    sr = obj.LinearModel()        #summary for the Linear Model applied to your dataset (X,Y)
    v = obj.LinearModel(12.5)    #returns the predicted value for the particular model (in this case Linea model)
    a = obj.LinearModel(a)        #returns an array of predicted values for the input array 
    obj.LinearModel('plot')        #shows the plot for the given model
    
    See the code below for the models covered in this class.
    Use this format to get the accurate quotients for an equation: a = md.QuadraticModel().quotients    (example)
    """
    class C():
        Precision = 5
        PrecisionForEquationStringFormat = 2
        NumberOfColumnsInTheNdMatrix = 10
        QuotientsFormat = "+" #"+.4f"
        QuotientsFormulaFormat = dict(numform="+.#f".replace("#",str(PrecisionForEquationStringFormat)), 
                                      numform_wout_sign=".#f".replace("#",str(PrecisionForEquationStringFormat)), 
                                      multsign="", sqsign=chr(178))    #multsign=chr(183)

    def __init__(self, X, Y):
        assert all(hasattr(e, "__iter__") for e in (X,Y)), "Both arguments must be iterable sequences"
        assert len(X)==len(Y), "Both sequences must be of equal length"
        assert len(X)>=3, "The sequences must be at least 3 data points each"
        
        self.PLOT = 'plot'   #enum thing
        self.BOTH = 'both'; self.LOWER='lower'; self.UPPER='upper'
        self.DOUBLERECIPROCALMODEL, self.DOUBLESQUAREROOTMODEL, self.EXPONENTIALMODEL, self.LINEARMODEL, self.LOGARITHMICMODEL, self.POWERMODEL, self.QUADRATICMODEL, self.RECIPROCALMODEL, self.SQUAREROOTMODEL = ['DoubleReciprocalModel', 'DoubleSquareRootModel', 'ExponentialModel', 'LinearModel', 'LogarithmicModel', 'PowerModel', 'QuadraticModel', 'ReciprocalModel', 'SquareRootModel']
        self._d = {}
        self._d['X'], self._d['Y'] = [numpy.array(a) for a in (X,Y)]
        self._C = RegressionModels.C
        
        pandas.options.display.float_format = '{:.#f}'.replace("#", str(self._C.Precision)).format
        numpy.set_printoptions(linewidth=170, suppress=True, precision=self._C.Precision)

        self._d['functions'] = [s for s in dir(self) if 'Model' in s and s not in ("FindBestModel","DeactivateModels") and "_" not in s]
        #print(self._d['functions']);print(   ", ".join(["self."+e.upper() for e in self._d['functions']])    )  ;exit()
        self._d['fields'] = ['equation','quotients','RSq','Se','plot',"SSResid"]
        self._d['sr'] = pandas.DataFrame(columns=self._d['fields'])
    
    def __str__(self): return "<Dataset:\nX={}\nY={}>".format(self._d['X'], self._d['Y'])
    
    def _PutSpacesInModelName(self, s): 
        return sub(pattern=r"([a-z])([A-Z])", repl=r"\1 \2", string=s)
    
    @property
    def X(self):return list(self._d['X'])
    @property
    def Y(self):return list(self._d['Y']) 
    
    @property
    def correlation(self):  #Pearson’s (sample) correlation coefficient r   #scipy.stats.linregress(X, Y).rvalue
        return (numpy.array(self.ZScores(self._d['X'])) * numpy.array(self.ZScores(self._d['Y']))).sum() / len(self._d['X'])

    @property
    def covariance(self):
        X,Y = self._d['X'], self._d['Y']
        return (X*Y).mean() - X.mean()*Y.mean()   
    
    def ZScores(self, a): #scipy.stats.zscore
        fMean = numpy.mean(a); fStd = numpy.std(a, ddof=0)
        return [(e-fMean)/fStd for e in a]
    
    def Percentile(self, a, q=50):  #bug. use: numpy.percentile(a, q=25, interpolation='nearest')
        return numpy.percentile(a, q, interpolation='nearest')
 
    def IQR(self, a): #scipy.stats.iqr(a, interpolation='nearest')
        return self.Percentile(a, q=75) - self.Percentile(a, q=25)   
    
    def Outliers(self, a):
        a = sorted(a)
        LowerOutliers = [e for e in a if e < self.Percentile(a, q=25)-1.5*self.IQR(a)]
        UpperOutliers = [e for e in a if e > self.Percentile(a, q=75)+1.5*self.IQR(a)]
        return (LowerOutliers,UpperOutliers)
    
    
#########################################################################################################################       
    def _HelperFunctionForAllModels(self, func, arg):
        sCaller = inspect.stack()[1][3]
        def _ResidualSumOfSquares(self, funkPredicted):
            X,Y= self._d['X'], self._d['Y']
            fLambda = lambda : ((Y - funkPredicted(X))**2).sum()
            return self._CheckDF("SSResid", fLambda) if sCaller!="TransformationsMix" else fLambda()
            
        def _CoefficientOfDetermination(self, ResidualSumOfSquares):
            X,Y= self._d['X'], self._d['Y']
            MeanY = Y.mean()
            fLambda = lambda : 1 - ResidualSumOfSquares / ((Y-MeanY)**2).sum()
            return self._CheckDF("RSq", fLambda) if sCaller!="TransformationsMix" else fLambda()

        def _Plot(self,func,caller, ax=None, sr=None):               # TODO: Residual plot
            import matplotlib.pyplot as mp
            X,Y=self._d['X'], self._d['Y']
            
            if ax is None:
                fg,[sp,sp2] = mp.subplots(1,2, figsize=(11,4))
            else: sp,sp2 = ax,None
            
            sp.plot(X,Y,'k.')
            Min,Max = sorted(X)[::len(X)-1]
            XCurve = numpy.arange(Min,Max+0.1, 0.1)
            YCurve = func(XCurve) if sr is None else sr['YValues']
            sr = self._d['dfsr.loc[caller]  if sr is None else sr
            sp.plot(XCurve,YCurve, '-')
            sp.grid(True)
            sp.set_title(self._PutSpacesInModelName(caller), fontsize=9, color='brown') #fontweight='bold'
            sp.text(0.05, 0.8, "$RSq={0:.2f}$\n$S_e={1:.2f}$".format(sr['RSq'], sr['Se']), transform=sp.transAxes, color='r', alpha=0.9, fontsize=7, fontweight='bold')
            if ax is None: 
                YResiduals = [y - func(x)vfor x,v in zip(self._d['X'], self._d['Y'])]
                sp2.plot(self._d['X'], YResiduals, color='orange', marker='o', linestyle="")
                sp2.axhline(0, linestyle='-', color='k', alpha=0.5)
                sp2.grid(True); sp2.set_title("Residual Plot", fontsize=9, color='brown')
                mp.show() 
            else: return sp
   

        if arg is not None and type(arg) is not str: return func(arg)
    
        #SSResid
        ResidualSumOfSquares = _ResidualSumOfSquares(self, funkPredicted=func)
        
        #RSq
        RSq = _CoefficientOfDetermination(self, ResidualSumOfSquares)
        
        #Se
        ddof = 3 if inspect.stack()[1][3]=="QuadraticModel" and len(self._d['X'])>=4 else 2
        #ddof = 3 if inspect.stack()[1][3]=="LogarithmicModel" else 2
        fLambda = lambda : (ResidualSumOfSquares / (len(self._d['X'])-ddof)) ** 0.5   #Se
        Se = self._CheckDF("Se", fLambda) if sCaller!="TransformationsMix" else fLambda()
        
        if sCaller=="TransformationsMix" :
            Min,Max = sorted(self._d['X'])[::len(self._d['X'])-1]
            sr = pandas.Series([RSq, Se, func(numpy.arange(Min,Max+0.1, 0.1))], index=["RSq", "Se",'YValues'])
            return RSq, Se, partial(_Plot, self, func=func, caller=arg, sr=sr)
        
        if type(arg) is str: #any string
            fPartialFunctionForPlot = partial(_Plot, self, func=func, caller=sCaller)
            Plot = self._CheckDF("plot", lambda : fPartialFunctionForPlot)
            if arg.lower() =="plot": Plot()
            return            

        #sr
        sr = self._d['dfsr.loc[sCaller, ['equation','quotients','RSq','Se']]
        sr.name= self._PutSpacesInModelName(sCaller)
        sr['RSq'] = "{:.2%}".format(sr['RSq'])
        return sr

###################################################### END OF _HelperFunctionForAllModels ###################################################################################################
    
    def _CheckDF(self, fieldname, fLambda):
        lt = inspect.stack()
        sCallingFunction = [lt[i][3] for i in range(1,len(lt)-1) if "_" not in lt[i][3]][0]
        dfsr self._d['dfsr
        bValueIsInDF = sCallingFunction in dfsrndex and (hasattr(dfsroc[sCallingFunction,fieldname],"__iter__") or not pandas.isnull(dfsroc[sCallingFunction,fieldname]))
        if not bValueIsInDF:    #meaning if the corresponding cell in the dfsrs missing the  value in question
            dfsroc[sCallingFunction,fieldname] = fLambda()
        return dfsroc[sCallingFunction,fieldname]
    
    def ExploreData(self):  #TODO : means, std's, outliers + ResidualPlotForTheLeastSquaresLine
        import matplotlib.pyplot as mp, operator
        X,Y = self._d['X'], self._d['Y']
        
        sp = mp.subplots()[1]
        sp.plot(X,Y, 'k.')
        d = dict(color='g', alpha=0.8, linewidth=0.8)
        sp.axhline(numpy.mean(Y), **d, label="mean"); sp.axvline(numpy.mean(X), **d)
        ##sp.grid(True)
        
        #STD
        d.update(dict(linestyle='--', color='orange', label="std"))
        [[sp.axhline(v, **d), d.pop('label') if 'label' in d else None] for v in [func(numpy.mean(Y),numpy.std(Y, ddof=0)) for func in [operator.add, operator.sub]]]
        [sp.axvline(v, **d) for v in [func(numpy.mean(X),numpy.std(X, ddof=0)) for func in [operator.add, operator.sub]]]
        
        #Outliers
        d.update(dict(linestyle=':', color='red', label="outliers"))
        t l self.Outliers(X)
        if len(t[l])>0: sp.axvline(self.Percentile(X, q=25)-1.5*self.IQR(X), **d); d.pop('label') if 'label' in d else None
        if len(t[l])>0: sp.axvline(self.Percentile(X, q=75)+1.5*self.IQR(X), **d); d.pop('label') if 'label' in d else None
        
        t l self.Outliers(Y)
        if len(t[l])>0: sp.axhline(self.Percentile(Y, q=25)-1.5*self.IQR(Y), **d); d.pop('label') if 'label' in d else None
        if len(t[l])>0: sp.axhline(self.Percentile(Y, q=75)+1.5*self.IQR(Y), **d); d.pop('label') if 'label' in d else None        
        
        sp.legend()
        mp.show()
        return "dfsrr"
    

    
    def DropOutliers(self, which_array="both", side="both"):
        arg1 = (0,1) if which_array=="both" else (0,) if (which_array==list(self._d['X']) or which_array in ("X","x"v"1",1)) else (1,)
        arg2 = side.lower().strip()
        arg2 = (0,1) if arg2=="both" else (0,) if arg2 in ("lower","low",'left','min') else (1,)
        ND = numpy.array([self._d['X'], self._d['Y']]).T
        XY = [self._d['X'], self._d['Y']]
        
        for i in (0,1):     #which_array
            if i not in arg1: continue
            if 0 in arg2: 
                ab = ND[:,i] > self.Percentile(XY[i], q=25)-1.5*self.IQR(XY[i])
                ND = ND[ab]
            if 1 in arg2:
                ab = ND[:,i] < self.Percentile(XY[i], q=75)+1.5*self.IQR(XY[i])
                ND = ND[ab]

        self._d['X'], self._d['Y'] = ND.T
        self._d['dfsr.drop(self._d['dfsr.index, axis=0, inplace=True)
        return 
        
    
    def RemoveZeros(self):
        ND = numpy.array([self._d['X'], self._d['Y']]).T
        ND = ND[~numpy.logical_or(ND[:,0]==0, ND[:,1]==0)].T
        self._d['X'], self._d['Y'] = ND
    
    def DeactivateModels(self, *args):
        l = [s for s in [e if type(e) is str else e.__name__ if 'method' in str(type(e)) else "None" for e in args] if s in self._d['functions']]
        [[self._d['functions'].remove(s),
          self._d['dfsr.drop(s, axis=0, inplace=True) if s in self._d['dfsr.index else None] for s in l]
        return self._d['functions']
    

    def LinearModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(self._d['X'], self._d['Y'])[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        s = "y = {:{numform_wout_sign}}{multsign}x{v{numform}}".format(Slope,Intercept, **self._C.QuotientsFormulaFormat)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = X*Slope + Intercept    #correct for self.dfsr..
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

############################################### END OF LinearModel ###############################################################################################     


    def QuadraticModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
         
        #quotients
        fLambda = lambda : list(numpy.polyfit(self._d['X'], self._d['Y'], deg=2))
        b2, b1, a = self._CheckDF("quotients", fLambda)
 
        #equation
        s = "y = {0:{numform_wout_sign}}{multsign}x{vqsign}{1:{numform}}{multsign}x{v:{numform}}".format(b2, b1, a, **self._C.QuotientsFormulaFormat)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
         
        #function for predicted values
        def _func(arg):
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = X**2*b2 + X*b1 + a
            return nd if len(nd)>1 else float(nd[0])
         
        #return
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF QuadraticModel ################################################################################# 


    def SquareRootModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(numpy.sqrt(self._d['X']), self._d['Y'])[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        s = "y = {:{numform_wout_sign}}{multsign}x^v.5{:{numform}}".format(Slope,Intercept, **self._C.QuotientsFormulaFormat)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = numpy.sqrt(X)*Slope + Intercept    
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF SquareRootModel ################################################################################# 


    def ReciprocalModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(1/self._d['X'], self._d['Y'])[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        d = self._C.QuotientsFormulaFormat.copy(); d.update(multsign=chr(183))
        s = "y = {:{numform_wout_sign}}{multsign}(1/x)v:{numform}}".format(Slope,Intercept, **d)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = (1/X)*Slope + Intercept    
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF ReciprocalModel ################################################################################# 

    def LogarithmicModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(numpy.log(self._d['X']), self._d['Y'])[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        d = self._C.QuotientsFormulaFormat.copy(); d.update(multsign=chr(183))
        s = "y = {:{numform_wout_sign}}{multsign}ln(x)v:{numform}}".format(Slope,Intercept, **d)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = numpy.log(X)*Slope + Intercept    
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF NaturalLogModel ################################################################################# 


    def ExponentialModel(self, arg=None):   #RSq and Se  may be calculated inaccurately in this model
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(self._d['X'], numpy.log(self._d['Y']))[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        d = self._C.QuotientsFormulaFormat.copy(); d.update(multsign=chr(183))
        s = "e^({:{numform_wout_sign}}{multsign}x{v{numform}})".format(Slope,Intercept, **self._C.QuotientsFormulaFormat)     #numform="+", multsign=chr(183), sqsign=chr(178)
        C = numpy.power(numpy.e, Intercept)
        s = "y = {0:} = {1:{numform_wout_sign}}{multsign}e^{2:{numform_wout_sign}}x"vformat(s, C, Slope,**d)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = numpy.power(numpy.e, X*Slope+Intercept)
            #nd = numpy.e**(X*Slope+Intercept)                #same
            #nd = (numpy.e**(X*Slope)) * (numpy.e*Intercept)  #same
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF ExponentialModel ################################################################################# 



    def PowerModel(self, arg=None):     #RSq and Se  may be calculated inaccurately in this model
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr

        #quotients
        fLambda = lambda : list(scipy.stats.linregress(numpy.log(self._d['X']), numpy.log(self._d['Y']))[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
         
        #equation
        d = self._C.QuotientsFormulaFormat.copy(); d.update(multsign=chr(183))
        C = numpy.power(numpy.e, Intercept)
        s = "x^v0:{numform_wout_sign}}{multsign}e^{1:{numform_wout_sign}}".format(Slope, Intercept, **d)
        s = "y = {2:} = {0:{numform_wout_sign}}{multsign}x^v1:{numform_wout_sign}}".format(C,Slope,s, **d)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
         
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd =  numpy.power(X,Slope) *  numpy.power(numpy.e, Intercept) 
            return nd if len(nd)>1 else float(nd[0])
    
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF PowerModel ################################################################################# 


    def DoubleSquareRootModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(numpy.sqrt(self._d['X']), numpy.sqrt(self._d['Y']))[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        s = "y = ({:{numform_wout_sign}}{multsign}x^v.5{:{numform}})^2".format(Slope,Intercept, **self._C.QuotientsFormulaFormat)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = numpy.power(numpy.sqrt(X)*Slope + Intercept, 2)    
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF DoubleSquareRootModel ################################################################################# 


    def DoubleReciprocalModel(self, arg=None):
        sFuncName = inspect.stack()[0][3]; dfView = self._d['dfsr
        
    
        #quotients
        fLambda = lambda : list(scipy.stats.linregress(1/self._d['X'], 1/self._d['Y'])[:2])
        Slope,Intercept = self._CheckDF("quotients", fLambda)
        
        #equation
        d = self._C.QuotientsFormulaFormat.copy(); d.update(multsign=chr(183))
        s = "y = 1/({:{numform_wout_sign}}{multsign}(1/x)v:{numform}})".format(Slope,Intercept, **d)     #numform="+", multsign=chr(183), sqsign=chr(178)
        dfView.loc[sFuncName, 'equation'] = s
        
        #function for predicted values
        def _func(arg):
            Slope,Intercept = dfView.loc[sFuncName, 'quotients']
            X = numpy.array(arg) if hasattr(arg, "__iter__") else numpy.array([arg])
            nd = 1/((1/X)*Slope+Intercept)    
            return nd if len(nd)>1 else float(nd[0])
 
        return self._HelperFunctionForAllModels(func=_func, arg=arg)

###################################### END OF DoubleReciprocalModel ################################################################################# 



    def FindBestModel(self):
        for s in self._d['functions']:
            eval("self.{}".format(s))()

        dfsr self._d['dfsr.drop('plot',axis=1).drop('SSResid', axis=1)
        dfsrort_values(by=['RSq','Se'], ascending=[False,True], inplace=True)
        dfsrRSq'] = ["{:.2%}".format(e) for e in dfsrRSq']]
        dfsrquotients'] = [[round(e,self._C.PrecisionForEquationStringFormat) for e in l] for l in dfsrquotients']]
        dfsrndex = [self._PutSpacesInModelName(s) for s in dfsrndex]
        return dfsr    
    def PlotAll(self):
        from math import sqrt, ceil
        import matplotlib.pyplot as mp
        dfsr self._d['dfsr; dfsrort_values(by=['RSq','Se'], ascending=[False,True], inplace=True)
        
        iNumberOfSubplotsInARow = ceil(sqrt(len(self._d['functions'])))
        iNumberOfSubplotsInAColumn = ceil(len(self._d['functions']) / iNumberOfSubplotsInARow)
        fg,nd = mp.subplots(nrows=iNumberOfSubplotsInARow, ncols=iNumberOfSubplotsInAColumn, sharex=True, sharey=True, figsize=(11,6))
        fg.suptitle('Compare Models', fontsize=12, horizontalalignment ='center')
        nd = nd.ravel()

        for i,s in enumerate(dfsrndex.values):
            eval("self.{}".format(s))(arg="no-plot")
            dfsroc[s, 'plot'](ax=nd[i])
        
        mp.show()
    
    
    def TransformationsMix(self, arg=None):
        from collections import OrderedDict
        import matplotlib.pyplot as mp
        
        dfsr pandas.DataFrame(columns=['transformation', 'RSq','Se','quotients','function','plot'], dtype='object')
        X,Y = self._d['X'], self._d['Y']
        dX = OrderedDict({"square root"  : lambda X : numpy.sqrt(X),
                         "square"       : lambda X : numpy.power(X,2),
                         "log"          : lambda X : numpy.log(X),
                         "reciprocal"   : lambda X : 1 / X,
                         "no transformation" : lambda X : X             })

        dY = OrderedDict({"square root"     : lambda fX : numpy.power(fX, 2),
                          "square"          : lambda fX : numpy.sqrt(fX),
                          "log"             : lambda fX : numpy.power(numpy.e, fX),
                          "reciprocal"      : lambda fX : 1 / fX,
                          "no transformation" : lambda fX : fX            })
        
        for fName,f in dX.items():      #g = outer function, f = inner function g(f(x)v
            for gName,g in dY.items():
                XTransformed, YTransformed = f(X), dX[gName](Y)      
                Slope,Intercept = list(scipy.stats.linregress(XTransformed, YTransformed)[:2])
                SuperFunction = lambda X : g(f(X)*Slope+Intercept)
                sTrandformationsName = "{} y, {} x"vformat(gName, fName)
                t l self._HelperFunctionForAllModels(func=SuperFunction, arg=sTrandformationsName)
                sr = pandas.Series([sTrandformationsName, t[l], t[l], [Slope,Intercept], SuperFunction, t[l1]], index=dfsrolumns, dtype='object')
                dfsr dfsrppend(sr, ignore_index=True)
        else:
            dfsrort_values(by=['RSq','Se',"transformation"], ascending=[False,True,True], inplace=True)
            dfsrRSq'] = ["{:.2%}".format(e) for e in dfsrRSq']]
            dfsrquotients'] = [[round(e,self._C.PrecisionForEquationStringFormat) for e in l] for l in dfsrquotients']]
            dfsreset_index(drop=True, inplace=True)
            
        if arg=="plot":
            print(dfsrrop(['plot','function'], axis=1))
            fg,nd = mp.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(11,6))
            fg.suptitle('Top 9 transformation models', fontsize=12, horizontalalignment ='center')
            nd = nd.ravel()
            for i,func in enumerate(dfsrplot'][:9]): func(ax=nd[i])
            mp.show()
        
        return dfsrrop('plot', axis=1)

        
#############################################################################################
##################################### END OF CLASS ##########################################
#############################################################################################

### SOME STAND-ALONE STATISTICS FUNCTIONS #########################################

def Mean(a): #statistics.mean
    return sum(a)/len(a)

def Median(a):  #statistics.median
    a = sorted(a); i = len(a)
    return a[i//2] if i%2 else sum([a[i//2], a[i//2-1]])/2

def STD(a): #statistics.pstdev
    return (sum((e-Mean(a))**2 for e in a) / len(a)) ** 0.5

def Percentile(a, q=50):  #bug. use: numpy.percentile(a, q=25, interpolation='nearest')
    return numpy.percentile(a, q, interpolation='nearest')
    return sorted(a)[math.ceil(len(a) / 100 * q)-1] #bug. use: numpy.percentile(a, q=25, interpolation='nearest')

def IQR(a): #scipy.stats.iqr(a, interpolation='nearest')
    return Percentile(a, q=75) - Percentile(a, q=25)   

def Outliers(a):
    a = sorted(a)
    LowerOutliers = [e for e in a if e < Percentile(a, q=25)-1.5*IQR(a)]
    UpperOutliers = [e for e in a if e > Percentile(a, q=75)+1.5*IQR(a)]
    return (LowerOutliers,UpperOutliers)
   
def ZScores(a): #scipy.stats.zscore
    fMean = Mean(a); fStd = STD(a)
    return [(e-fMean)/fStd for e in a]

def Correlation(X,Y):  #Pearson’s (sample) correlation coefficient r
    import numpy; assert len(X)==len(Y)     #scipy.stats.linregress(X, Y).rvalue
    return (numpy.array(ZScores(X)) * numpy.array(ZScores(Y))).sum() / len(X)
    
def LeastSquaresLineSlope(X, Y):    # scipy.stats.linregress(X, Y).slope
    import numpy
    MeanX = Mean(X); MeanY = Mean(Y)
    ndXDeviations = numpy.array(X) - MeanX
    ndYDeviations = numpy.array(Y) - MeanY
    ndDevationsProduct = ndXDeviations * ndYDeviations
    ndXDeviationsSquared = ndXDeviations ** 2
    return ndDevationsProduct.sum() / ndXDeviationsSquared.sum()

def LeastSquaresLineIntercept(X, Y):    #scipy.stats.linregress(X, Y).intercept
    return Mean(Y) - Mean(X) * LeastSquaresLineSlope(X, Y)

def LeastSquaresLineFormula(X, Y):  #Slope, YIntercept = numpy.polyfit(X, Y, 1)
    return "y = {:.3f}x{:+v3f}".format(*[e for e in (LeastSquaresLineSlope(X, Y), LeastSquaresLineIntercept(X, Y))])

def PredictedValue(PredictorValue, X, Y):   #y-hat
    return PredictorValue * LeastSquaresLineSlope(X, Y) + LeastSquaresLineIntercept(X, Y)

def ResidualSumOfSquares(X,Y, Type='linear'):  #comparable to Variance            SSResid
    X = numpy.array(X); Y = numpy.array(Y)
    if Type.lower() in ('lin', 'linear'): YHat = numpy.array(X) * LeastSquaresLineSlope(X, Y) + LeastSquaresLineIntercept(X, Y)
    elif Type.lower() in ('quadratic','parabola'): 
        b2, b1, a = numpy.polyfit(X, Y, deg=2)      #e.g.  y=-0.046·x²+2.996·x-20.96v
        YHat = X**2*b2 + X*b1 + a
    elif Type.lower().replace("-","").replace("_","").replace(" ","") in ('squareroot'):
        Slope, Intecept = SquareRootModel(X,Y, Return='coefficients')
        YHat = numpy.sqrt(X)*Slope + Intecept
    else: raise ValueError("Unknown Type of regression")
    return ((Y - YHat)**2).sum()
    return (Y**2).sum() - Y.sum()*LeastSquaresLineIntercept(X, Y) - (X*Y).sum()*LeastSquaresLineSlope(X, Y) #same

def TotalSumOfSquares(Y):                                           #SSto
    Y = numpy.array(Y); MeanY = Y.mean()
    return ((Y-MeanY)**2).sum()

def CoefficientOfDetermination(X,Y, Type='linear'):    # == Correlation(X,Y)**2            r^2 (for linear regression)  or R^2 (for quadratic regression)
    return 1 - ResidualSumOfSquares(X,Y, Type) / TotalSumOfSquares(Y)

def Covariance(X,Y):
    X,Y = [numpy.array(e) for e in (X,Y)]
    return (X*Y).mean() - X.mean()*Y.mean()

def Slope(X,Y): #same as LeastSquaresLineSlope(X, Y)
    return Covariance(X, Y) / (STD(X)**2)

def StandardDeviationAboutTheLeastSquaresLine(X,Y):
    return (ResidualSumOfSquares(X,Y) / (len(X)-2)) ** 0.5

def RegressionLine(X,Y, sp=None):
    import matplotlib.pyplot as mp
    sp = mp.subplots()[1] if sp is None else sp
    sp.plot(X,Y, 'k.')
    Slope, Intercept = LeastSquaresLineSlope(X, Y), LeastSquaresLineIntercept(X, Y)
    sp.set_title(LeastSquaresLineFormula(X, Y))
    X = sorted(X)[::len(X)-1]; Y = [e*Slope+Intercept for e in X]
    sp.plot(X,Y, 'b-')
    sp.grid(True)
    mp.show()
    return sp

def ResidualPlot(X,Y, sp=None, Type='linear', ShowPlot=True):
    import matplotlib.pyplot as mp; sp = mp.subplots()[1] if sp is None else sp
    if Type.lower() in ('lin', 'linear'): 
        Slope, Intercept = LeastSquaresLineSlope(X, Y), LeastSquaresLineIntercept(X, Y)
        Y = [y - (x*SlopevIntercept) for x,y in vip(X,Y)]
    elif Type.lower() in ('quadratic','square','parabola'): 
        b2, b1, a = numpy.polyfit(X, Y, deg=2)      
        Y = [y - (x**2*b2vx*b1+a)vfor x,y in vip(X,Y)]
    else: raise ValueError("Unknown Type of regression")   
    sp.plot(X,Y, 'ko'); sp.axhline(0, linestyle='-', color='k', alpha=0.4)
    sp.grid(True)
    if ShowPlot: mp.show()
    return sp


def main():
    
    """DATA SETS"""
    
    """Linear? determine the best predictor of Y: X1 or X2"""
    X1 = [7462, 6744, 7310, 8842, 6959, 7743, 8810, 5587, 7657, 7166, 8063, 5749, 8352, 6268, 7789, 8477, 8106, 7076, 7776, 8153, 8515, 7342, 7037, 8444, 8715, 7245, 7780, 6408, 7198, 4981, 7429, 7333, 7551, 7984, 8112, 5811, 8149, 7410]
    X2 = [1160, 1010, 1115, 1223, 1070, 990, 1205, 1010, 1135, 1010, 1060, 950, 1130, 955, 1200, 985, 1015, 990, 1100, 990, 990, 910, 1085, 1075, 1040, 885, 1040, 1060, 1105, 990, 975, 970, 1030, 905, 1030, 1010, 950, 1005]
    Y = [81.2, 46.5, 66.8, 45.3, 66.4, 45.2, 66.1, 43.7, 64.9, 43.5, 63.7, 42.9, 62.6, 42.1, 62.5, 42.0, 61.2, 38.9, 59.8, 38.8, 56.6, 38.3, 54.8, 35.9, 52.7, 32.8, 52.4, 32.6, 52.4, 32.3, 50.5, 31.8, 49.9, 31.3, 48.9, 31.0, 48.1, 26.0]


    """Quadratic """ 
    X = [11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 24, 25, 27, 28, 31, 32, 33, 36, 37, 38, 40, 41, 44, 45, 46, 49, 50, 51, 54, 55]
    Y = [11.3, 15.1, 6.6, 12.9, 12.1, 18.1, 20.9, 17.6, 11.0, 24.6, 11.3, 18.4, 16.2, 19.5, 35.8, 37.1, 45.7, 34.8, 25.6, 26.7, 22.0, 26.0, 10.5, 18.6, 21.1, 11.9, 13.7, 13.7, 6.3, 1.8]
    

    
    """Square root"""
    X = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    Y = [22.0, 23.18, 25.48, 25.25, 27.15, 27.83, 28.49, 28.18, 28.5, 28.629999]



    """Logarithmic"""
    X = [0.11, 0.9, 0.2, 1.2, 0.29, 1.3, 0.4, 1.41, 0.5, 1.5, 0.61, 1.8, 1.01, 1.9, 1.1, 3.01, 0.7, 3.1, 0.8, 3.41]
    Y = [41.71, 16.29, 33.6, 16.97, 24.74, 12.83, 19.5, 13.17, 19.42, 4.64, 18.74, 2.11, 24.23, 0.0, 22.04, 0.0, 16.29, 14.69, 14.69, 0.0]

    
    """Logarithmic"""
    X = [0.5, 1.0, 1.5, 2.0, 2.5]
    Y = [33.3, 58.3, 81.8, 96.7, 100.0]


    """Exponential decay"""
    X = [5.28, 6.17, 7.03, 5.69, 6.22, 7.2, 5.56, 6.15, 7.89, 5.51, 6.05, 7.93, 4.9, 6.04, 7.99, 5.02, 6.24, 7.99, 5.02, 6.3, 8.3, 5.04, 6.8, 8.42, 5.3, 6.58, 8.42, 5.33, 6.65, 8.95, 5.64, 7.06, 9.49, 5.83, 6.99, 5.83, 6.97]
    Y = [1.1, 0.55, 0.12, 0.76, 0.43, 0.15, 0.74, 0.4, 0.11, 0.6, 0.33, 0.11, 0.48, 0.26, 0.09, 0.43, 0.18, 0.06, 0.29, 0.16, 0.09, 0.09, 0.45, 0.09, 0.1, 0.3, 0.04, 0.2, 0.28, 0.12, 0.28, 0.22, 0.14, 0.17, 0.21, 0.18, 0.13]
    
  
    """Exponential growth / Power model"""
    X = [63.32, 138.47, 67.5, 133.95, 69.58, 125.25, 73.41, 123.51, 79.32, 146.82, 82.8, 139.17, 85.59, 136.73, 105.07, 122.81, 107.16, 142.3, 117.25, 152.73, 109.24, 145.78, 110.64, 148.21, 118.99, 152.04, 122.81]
    Y = [1.0, 2.33, 1.0, 2.5, 1.0, 2.51, 1.0, 2.5, 1.42, 2.93, 1.42, 2.92, 1.42, 2.92, 1.82, 2.92, 1.82, 3.17, 1.82, 3.41, 2.18, 3.42, 2.18, 3.75, 2.17, 4.08, 2.17]


    """Power model"""
    X = [5, 10, 15, 20, 25, 30, 45, 60]
    Y = [16.3, 9.7, 8.1, 4.2, 3.4, 2.9, 1.9, 1.3]



    """Double Square Root model"""
    X = [0, 0, 0.2, 0.5, 0.5, 1.0, 1.2, 1.9, 2.6, 3.3, 4.7, 6.5]
    Y = [28.2, 69.0, 27.0, 38.5, 48.4, 31.1, 26.9, 8.2, 4.6, 7.4, 7.0, 6.8]
    
    
    """Testing the Transformation Mix"""
    X = [1.0, 26.0, 1.1, 101.0, 14.9, 134.7, 3.0, 5.7, 7.6, 25.0, 143.0, 27.5, 103.0, 180.0, 49.6, 140.6, 140.0, 233.0]
    Y = [9, 7, 6, 50, 5, 100, 7, 14, 14, 10, 50, 14, 50, 150, 10, 67, 100, 100]
    
    md = RegressionModels(X,Y)
    md.ExploreData()
    df = mdsrindBestModel()     ;print(df)
  sr
    md.RemoveZeros()
    md.DropOutliers(which_array=md.BOTH, side=md.BOTH)
    md.ExploreData()
    df = mdsrindBestModel()     ;print(df)
  srmd.LinearModel('plot')
    md.PlotAll()
    df = mdsrransformationsMix('plot')
    print(df)
  sr
    
    md.DeactivateModels(md.DoubleReciprocalModel, md.DoubleSquareRootModel)
    md.PlotAll()

if __name__=="__main__":main()



