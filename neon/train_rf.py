#!/usr/bin/env python

"""
Examples:

    python jobwait_mlp.py -b gpu -e 10
        Run the example for 10 epochs of mnist data using the nervana gpu backend

    python jobwait_mlp.py --validation_freq 1
        After each training epoch the validation/test data set will be processed through the model
        and the cost will be displayed.

    python jobwait_mlp.py --serialize 1 -s checkpoint.pkl
        After every iteration of training the model will be dumped to a pickle file named
        "checkpoint.pkl".  Changing the serialize parameter changes the frequency at which the
        model is saved.

    python jobwait_mlp.py --model_file checkpoint.pkl
        Before starting to train the model, the model state is set to the values stored in the
        checkpoint file named checkpoint.pkl.
"""

import logging
import csv
import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor as rfr

def smoothL1(x,y):
    t=x-y;
    return np.mean((0.5 * np.square(t) * (np.absolute(t) < 1) +
           (np.absolute(t) - 0.5) * (np.absolute(t) >= 1)))


#preprocessor
std_scale = preprocessing.StandardScaler(with_mean=True,with_std=True)


#number of non one-hot encoded features, including ground truth
num_feat=11

# load up the mnist data set
# split into train and tests sets
#load data from csv-files and rescale
#training
traindf=pd.DataFrame.from_csv('../csv/cori_data_train.csv')
ncols=traindf.shape[1]
tmpmat=traindf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.fit_transform(tmpmat[:,1:num_feat])
X_train=tmpmat[:,1:]
y_train=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#validation
validdf=pd.DataFrame.from_csv('../csv/cori_data_validate.csv')
ncols=validdf.shape[1]
tmpmat=validdf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.transform(tmpmat[:,1:num_feat])
X_valid=tmpmat[:,1:]
y_valid=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#test
testdf=pd.DataFrame.from_csv('../csv/cori_data_test.csv')
ncols=testdf.shape[1]
tmpmat=testdf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.transform(tmpmat[:,1:num_feat])
X_test=tmpmat[:,1:]
y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#fit random forest
cpred = rfr(n_estimators=100, max_features="log2")
cpred = cpred.fit(X_train, np.ravel(y_train))

#predict test set:
y_pred=cpred.predict(X_test)

#evaluate model
print('Evaluation Error = %.4f'%smoothL1(y_pred,y_test))
