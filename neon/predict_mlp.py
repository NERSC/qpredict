import os
import logging
import json
import os
import re
import pandas as pd
import numpy as np

from neon.callbacks.callbacks import Callbacks
from neon.data.dataiterator import ArrayIterator
from neon.initializers import Xavier
from neon.layers import GeneralizedCost, Affine, Linear, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Adam, ExpSchedule
from neon.transforms import Rectlin#, MeanSquared
from neon.util.argparser import NeonArgparser
from cost import MeanSquaredLoss, MeanSquaredMetric, SmoothL1Loss, SmoothL1Metric

#data handler
import data

#init model function
def init_model():
    #load the preprocessor
    npzfile = np.load('jobwait_preproc.npz')
    mean = npzfile['mean']
    std = npzfile['std']
    mean = np.reshape(mean, (1,mean.shape[0]))
    std = np.reshape(std, (1,std.shape[0]))
    
    #neon insanity
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()
    
    #load the model
    mlp=Model("jobwait_model.prm")
    return mlp,mean,std
    

#actual prediction routine
def predict_mlp(inputdf):
    #number of non-one-hot features including label, init model
    num_feat = 9
    mlp,mean,std=init_model()

    #preprocess
    X_pred=inputdf.as_matrix()
    X_pred[:,:num_feat] -= mean
    X_pred[:,:num_feat] /= std
    pred_set = ArrayIterator(X=X_pred, make_onehot=False)

    print X_pred.shape

    #predict and return
    return mlp.get_outputs(pred_set)


#main function, so that it can be called directly for debugging
def main():
    #load data and preprocess
    testdf=pd.DataFrame.from_csv('../csv/cori_data_test.csv')
    
    #remove one-hot: just for debugging. First, determine which columns are one-hot encoded:
    ohcols=[]
    for feature in data.onehotfeaturelist:
        ohcols+=[x for x in testdf.columns if re.match(feature+'_[0-9].*',x)]
    
    #un-one-hot-encode: get supercategories:
    supercats=list(set([x.split('_')[0] for x in ohcols]))
    numclasses=[]
    for cat in supercats:
        numclasses.append(len([x for x in ohcols if cat in x]))
    
    for idx,feature in enumerate(supercats):
        testdf[feature+'_tag']=0.
        for index in range(numclasses[idx]):
            testdf[feature+'_tag']+=testdf[feature+'_'+str(index)]*index
            del testdf[feature+'_'+str(index)]
        testdf[feature+'_tag']=testdf[feature+'_tag'].apply(lambda x: str(int(x)))
        
    #convert to json
    json_data=testdf.to_json(orient='records', double_precision=10)
    
    #create data frame with correct feature ordering, one-hot encoding:
    tmpdf=data.create_df_from_json(json_data,True)
    
    #convert dataframe to json object, redo one-hot:
    print predict_mlp(tmpdf)
    

#call main
if __name__ =='__main__':
    main()