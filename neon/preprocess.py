import numpy as np

class feature_scaler(object):
    
    def __init__(self, type='Standardizer', **kwargs):
        self.type=type
        self.axis=0 #only support first axis for the moment
        self.fitted=False
        
        if self.type=='Standardizer':
            if 'with_mean' in kwargs:
                self.with_mean=kwargs['with_mean']
            else:
                self.with_mean=True
                
            if 'with_std' in kwargs:
                self.with_std=kwargs['with_std']
            else:
                self.with_std=True
                
        elif self.type=='RobustStandardizer':
            if 'with_median' in kwargs:
                self.with_mean=kwargs['with_median']
            else:
                self.with_mean=True
                
            if 'with_central68' in kwargs:
                self.with_std=kwargs['with_central68']
            else:
                self.with_std=True
        
        else:
            raise ValueError('Unknown scaler type')
    
    
    def fit(self, X):
        if self.type=='Standardizer':
            self.meanvals=np.mean(X,axis=self.axis)
            self.stdvals=np.std(X,axis=self.axis)
        elif self.type=='RobustStandardizer':
            self.meanvals=np.median(X,axis=self.axis)
            tmplo=np.percentile(X,15.87,axis=self.axis)
            tmphi=np.percentile(X,84.13,axis=self.axis)
            self.stdvals=(tmphi-tmplo)/2.
        
        if not self.with_mean:
            self.meanvals=np.zeros((1,X.shape[1]))
        if not self.with_std:
            self.stdvals=np.ones((1,X.shape[1]))
            
        #set std-values which are zero to one:
        self.stdvals[self.stdvals==0.]=1.
        self.fitted=True
    
    def transform(self, X):
        result=X
        if not self.fitted:
            print 'Please fit the scaler first!'
            return result
        result=X-np.tile(self.meanvals,(X.shape[0],1))
        result/=np.tile(self.stdvals,(X.shape[0],1))
        return result
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    
        