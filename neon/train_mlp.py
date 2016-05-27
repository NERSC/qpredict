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
#from custom_dataiterator import CustomDataIterator
from neon.data.dataiterator import ArrayIterator
from neon.initializers import Xavier
from neon.layers import GeneralizedCost, Affine, Linear, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Adam, ExpSchedule
from neon.transforms import Rectlin, Explin, Tanh#, MeanSquared
from neon.util.argparser import NeonArgparser
from sklearn import preprocessing
#from preprocess import feature_scaler #Thorsten's custom preprocessor
from cost import MeanSquaredLoss, MeanSquaredMetric, SmoothL1Loss, SmoothL1Metric, SmoothRobustLoss


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
#dataset
parser.add_argument('--training_set',type=str,help='csv file with training data')
parser.add_argument('--validation_set',type=str,help='csv file with validation data')
parser.add_argument('--test_set',type=str,help='csv file with test data')
#algorithmic parameters
parser.add_argument('--initial_learning_rate',type=float,default=1.e-4,help='initial learning rate for the optimizer')
parser.add_argument('--hidden_size',type=int,default=-1,help='size of the hidden layer. -1 means that it is set to the size of the input layer')
parser.add_argument('--keep_probability',type=float,default=0.5,help='keep probability in dropout layer')
parser.add_argument('--activation_function',type=str,default='ReLU',help='activation function used: supports ReLU, ELU, Tanh')
parser.add_argument('--checkpoint_restart',type=str,default=None,help='restart training from checkpoint')
parser.add_argument('--reset_epochs',action="store_true",help='decide whether to reset number of trained epochs to zero when checkpoint restart is used')
#output directory
parser.add_argument('--output_dir',type=str,help='folder to which the output will be stored')
args = parser.parse_args()
print "Training set: ", args.training_set
print "Validation set: ",args.validation_set
print "Test set: ",args.test_set
print "Output set: ",args.output_dir
#parameters
print "Initial learning rate: ",args.initial_learning_rate
print "Hidden Size: ",args.hidden_size
print "Keep probability: ",args.keep_probability
print "Activation Function: ",args.activation_function

if args.checkpoint_restart:
    print "Restarting from file: ",args.checkpoint_restart
    if args.reset_epochs:
        print "resetting epoch counter."

logger = logging.getLogger()
logger.setLevel(args.log_thresh)

# hyperparameters
num_epochs = args.epochs

#preprocessor
std_scale = preprocessing.StandardScaler(with_mean=True,with_std=True)
#std_scale = feature_scaler(type='Standardizer',with_mean=True,with_std=True)

#number of non one-hot encoded features, including ground truth
num_feat=11
#we have removed workAhead for testing if it makes a difference
#num_feat=10

# load up the mnist data set
# split into train and tests sets
#load data from csv-files and rescale
#training
traindf=pd.DataFrame.from_csv(args.training_set)

#remove workAhead feature in order to test its influence:
#del traindf['workAhead']
ncols=traindf.shape[1]

tmpmat=traindf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.fit_transform(tmpmat[:,1:num_feat])
print std_scale.scale_
print std_scale.mean_

tmpmat=traindf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.fit_transform(tmpmat[:,1:num_feat])
X_train=tmpmat[:,1:]
y_train=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#validation
validdf=pd.DataFrame.from_csv(args.validation_set)
#del validdf['workAhead']
ncols=validdf.shape[1]
tmpmat=validdf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.transform(tmpmat[:,1:num_feat])
X_valid=tmpmat[:,1:]
y_valid=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#test
testdf=pd.DataFrame.from_csv(args.test_set)
#del testdf['workAhead']
ncols=testdf.shape[1]
tmpmat=testdf.as_matrix()
tmpmat[:,1:num_feat]=std_scale.transform(tmpmat[:,1:num_feat])
X_test=tmpmat[:,1:]
y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))


# setup a training set iterator
#train_set = CustomDataIterator(X_train, lshape=(X_train.shape[1]), y_c=y_train)
train_set = ArrayIterator(X_train, lshape=(X_train.shape[1]), y=y_train, make_onehot=False)
# setup a validation data set iterator
#valid_set = CustomDataIterator(X_valid, lshape=(X_valid.shape[1]), y_c=y_valid)
valid_set = ArrayIterator(X_valid, lshape=(X_valid.shape[1]), y=y_valid, make_onehot=False)
# setup a validation data set iterator
#test_set = CustomDataIterator(X_test, lshape=(X_test.shape[1]), y_c=y_test)
test_set = ArrayIterator(X_test, lshape=(X_test.shape[1]), y=y_test, make_onehot=False)

# setup weight initialization function
init_norm = Xavier()

# setup model layers
actfunc=None
if args.activation_function=='ReLU':
	actfunc=Rectlin()
elif args.activation_function=='ELU':
    actfunc=Explin()
elif args.activation_function=='Tanh':
    actfunc=Tanh()
else:
	raise ValueError('activation_function invalid: please specify either ReLU, ELU, Tanh')

#topology:
if args.hidden_size<0:
    args.hidden_size=2*X_train.shape[1]+1
    
layers = [Affine(nout=args.hidden_size, init=init_norm, activation=actfunc),
          Dropout(keep=args.keep_probability),
          Linear(nout=1, init=init_norm)]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=SmoothL1Loss())
#cost = GeneralizedCost(costfunc=SmoothRobustLoss(beta=1.e-5))

# setup optimizer
#schedule
#schedule = ExpSchedule(decay=0.3)
#optimizer = GradientDescentMomentum(0.0001, momentum_coef=0.9, stochastic_round=args.rounding, schedule=schedule)
optimizer = Adam(learning_rate=args.initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.e-8)

# initialize model object: restart from checkpoint if necessary
if not args.checkpoint_restart:
    mlp = Model(layers=layers)
else:
    mlp = Model(args.checkpoint_restart)
    if args.reset_epochs:
        mlp.epoch_index=0

# configure callbacks
if args.callback_args['eval_freq'] is None:
	args.callback_args['eval_freq'] = 1

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

callbacks.add_early_stop_callback(stop_func)
callbacks.add_save_best_state_callback(os.path.join(args.data_dir, "early_stop-best_state.pkl"))

# run fit
mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#evaluate model
valid_error=mlp.eval(valid_set, metric=SmoothL1Metric())
print('Evaluation Error = %.8f'%(mlp.eval(valid_set, metric=SmoothL1Metric())))
print('Validation set error = %.8f'%(valid_error))

# Saving the model
print 'Saving model parameters!'
mlp.save_params(output_dir+'/jobwait_model.prm')

# Reloading saved model
# This should go in run.py
mlp=Model(output_dir+'/jobwait_model.prm')
print('Test set error = %.8f'%(mlp.eval(test_set, metric=SmoothL1Metric())))

# save the preprocessor vectors:
np.savez(output_dir+'/jobwait_preproc', mean=std_scale.mean_, std=std_scale.scale_)

#return error on validation set (for using it with spearmint)
#return valid_error
