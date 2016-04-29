import os
import logging
import json
import os
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
    num_feat = 8
    mlp,mean,std=init_model()

    #preprocess
    X_pred=inputdf.as_matrix()
    X_pred[:,:num_feat] -= mean
    X_pred[:,:num_feat] /= std
    pred_set = ArrayIterator(X=X_pred, make_onehot=False)

    #predict and return
    return mlp.get_outputs(pred_set)


#main function, so that it can be called directly for debugging
def main():
    #load data and preprocess
    testdf=pd.DataFrame.from_csv('../csv/cori_data_test.csv')
    
    #remove one-hot
    json_data=testdf.to_json(orient='records', double_precision=10)
    
    #create data frame with correct feature ordering, one-hot encoding:
    tmpdf=create_df_from_json(json_data)
    
    #convert dataframe to json object, redo one-hot:
    print predict_mlp(tmpdf)
    

#call main
if __name__ =='__main__':
    main()