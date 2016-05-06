import os
import logging
import csv
import os
import pandas as pd
import numpy as np

from neon.callbacks.callbacks import Callbacks
from custom_dataiterator import CustomDataIterator
from neon.initializers import Xavier
from neon.layers import GeneralizedCost, Affine, Linear, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Adam, ExpSchedule
from neon.transforms import Rectlin#, MeanSquared
from neon.util.argparser import NeonArgparser
from sklearn import preprocessing
#from preprocess import feature_scaler #Thorsten's custom preprocessor
from cost import MeanSquaredLoss, MeanSquaredMetric, SmoothL1Loss, SmoothL1Metric

parser = NeonArgparser(__doc__)

args = parser.parse_args()

num_feat = 9

npzfile = np.load('jobwait_preproc.npz')
mean = npzfile['mean']
std = npzfile['std']
mean = np.reshape(mean, (1,mean.shape[0]))
std = np.reshape(std, (1,std.shape[0]))

#preprocessor
#std_scale = preprocessing.StandardScaler(with_mean=True,with_std=True)
#std_scale.mean_ = mean
#std_scale.scale_ = std 

#traindf=pd.DataFrame.from_csv('cori_data_train.csv')
#ncols=traindf.shape[1]
#tmpmat=std_scale.fit_transform(traindf.as_matrix())

#test
testdf=pd.DataFrame.from_csv('../csv/cori_data_test.csv')
ncols=testdf.shape[1]
tmpmat=testdf.as_matrix()
tmpmat[:,:num_feat] -= mean
tmpmat[:,:num_feat] /= std  

X_test=tmpmat[:,1:]
y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

# setup a validation data set iterator
test_set = CustomDataIterator(X=X_test, lshape=(X_test.shape[1]), y_c=y_test)

# Reloading saved model
mlp=Model("jobwait_model.prm")
print('Test set error = %.4f'%(mlp.eval(test_set, metric=SmoothL1Metric())))

