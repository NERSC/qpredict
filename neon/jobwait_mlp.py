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
#from preprocess import feature_scaler
from cost import MeanSquaredLoss, MeanSquaredMetric, SmoothL1Loss, SmoothL1Metric


# Stop if validation error ever increases from epoch to epoch
def stop_func(s, v):
	if s is None:
		return (v, False)
	return (min(v, s), v > s)


# TODO Most of this file should be made a module called training.py
# To be called from main.py
# parse the command line arguments
# TODO This needs to be called in main with args passed to training
parser = NeonArgparser(__doc__)

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_thresh)

# hyperparameters
num_epochs = args.epochs

#preprocessor
std_scale = preprocessing.StandardScaler(with_mean=True,with_std=True)
#std_scale = feature_scaler(type='Standardizer',with_mean=True,with_std=True)

#number of non one-hot encoded features, including ground truth
num_feat=9

# load up the mnist data set
# split into train and tests sets
#load data from csv-files and rescale
#training
traindf=pd.DataFrame.from_csv('cori_data_train.csv')
ncols=traindf.shape[1]

tmpmat=std_scale.fit_transform(traindf.as_matrix())
print std_scale.scale_
print std_scale.mean_

tmpmat=traindf.as_matrix()
tmpmat[:,:num_feat]=std_scale.fit_transform(tmpmat[:,:num_feat])
X_train=tmpmat[:,1:]
y_train=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#validation
validdf=pd.DataFrame.from_csv('cori_data_validate.csv')
ncols=validdf.shape[1]
tmpmat=validdf.as_matrix()
tmpmat[:,:num_feat]=std_scale.transform(tmpmat[:,:num_feat])
X_valid=tmpmat[:,1:]
y_valid=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#test
testdf=pd.DataFrame.from_csv('cori_data_test.csv')
ncols=testdf.shape[1]
tmpmat=testdf.as_matrix()
tmpmat[:,:num_feat]=std_scale.transform(tmpmat[:,:num_feat])
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
#schedule
#schedule = ExpSchedule(decay=0.3)
#optimizer = GradientDescentMomentum(0.0001, momentum_coef=0.9, stochastic_round=args.rounding, schedule=schedule)
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1.e-8)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
print dir(args)

if args.callback_args['eval_freq'] is None:
	args.callback_args['eval_freq'] = 1

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

#callbacks.add_early_stop_callback(stop_func)
#callbacks.add_save_best_state_callback(os.path.join(args.data_dir, "early_stop-best_state.pkl"))

callbacks.add_early_stop_callback(stop_func)
callbacks.add_save_best_state_callback(os.path.join(args.data_dir, "early_stop-best_state.pkl"))

# run fit
mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#evaluate model
print('Evaluation Error = %.4f'%(mlp.eval(valid_set, metric=SmoothL1Metric())))
print('Test set error = %.4f'%(mlp.eval(test_set, metric=SmoothL1Metric())))

# Saving the model
print 'Saving model parameters!'
mlp.save_params("jobwait_model.prm")

# Reloading saved model
# This should go in run.py
mlp=Model("jobwait_model.prm")
print('Test set error = %.4f'%(mlp.eval(test_set, metric=SmoothL1Metric())))

# save the model
print 'Saving model parameters!'
mlp.save_params("jobwait_model.prm")

# save the preprocessor vectors:
np.savez("jobwait_preproc.prm", mean=std_scale.meanvals, std=std_scale.stdvals)
