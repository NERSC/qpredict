import os
import logging
import csv
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

parser = NeonArgparser(__doc__)

args = parser.parse_args()

num_feat = 9

#load the preprocessor
npzfile = np.load('jobwait_preproc.npz')
mean = npzfile['mean']
std = npzfile['std']
mean = np.reshape(mean, (1,mean.shape[0]))
std = np.reshape(std, (1,std.shape[0]))

#load data and preprocess
testdf=pd.DataFrame.from_csv('../csv/cori_data_test.csv')
ncols=testdf.shape[1]

#preprocess
tmpmat=testdf.as_matrix()
tmpmat[:,1:num_feat] -= mean
tmpmat[:,1:num_feat] /= std

X_pred=tmpmat[:,1:]
#y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))
pred_set = ArrayIterator(X=X_pred, make_onehot=False)

# Reloading saved model
mlp=Model("jobwait_model.prm")
Y_out=mlp.get_outputs(pred_set)

print Y_out

