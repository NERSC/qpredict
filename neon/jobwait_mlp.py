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
import pandas as pd
import numpy as np

from neon.callbacks.callbacks import Callbacks
from custom_dataiterator import CustomDataIterator
from neon.initializers import Xavier
from neon.layers import GeneralizedCost, Affine, Linear, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Adam
from neon.transforms import Rectlin#, MeanSquared
from neon.util.argparser import NeonArgparser
from sklearn import preprocessing
from cost import MeanSquared, SmoothL1Loss


# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_thresh)

# hyperparameters
num_epochs = args.epochs

#preprocessor
std_scale = preprocessing.StandardScaler(with_mean=True,with_std=True)

# load up the mnist data set
# split into train and tests sets
#load data from csv-files and rescale
#training
traindf=pd.DataFrame.from_csv('cori_data_train.csv')
ncols=traindf.shape[1]
tmpmat=std_scale.fit_transform(traindf.as_matrix())
X_train=tmpmat[:,1:]
y_train=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#validation
validdf=pd.DataFrame.from_csv('cori_data_validate.csv')
ncols=validdf.shape[1]
tmpmat=std_scale.transform(validdf.as_matrix())
X_valid=tmpmat[:,1:]
y_valid=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#test
testdf=pd.DataFrame.from_csv('cori_data_test.csv')
ncols=testdf.shape[1]
tmpmat=std_scale.transform(testdf.as_matrix())
X_test=tmpmat[:,1:]
y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))


# setup a training set iterator
train_set = CustomDataIterator(X_train, lshape=(X_train.shape[1]), y_c=y_train)
# setup a validation data set iterator
valid_set = CustomDataIterator(X_valid, lshape=(X_valid.shape[1]), y_c=y_valid)
# setup a validation data set iterator
test_set = CustomDataIterator(X_test, lshape=(X_test.shape[1]), y_c=y_test)

# setup weight initialization function
init_norm = Xavier()

# setup model layers
layers = [Affine(nout=X_train.shape[1], init=init_norm, activation=Rectlin()),
          Dropout(keep=0.5),
          Linear(nout=1, init=init_norm)]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=SmoothL1Loss())

# setup optimizer
#optimizer = GradientDescentMomentum(0.0001, momentum_coef=0.9, stochastic_round=args.rounding)
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, train_set, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#print('L2-error = %.4f%%' % (mlp.eval(valid_set, metric=SumSquared())))
