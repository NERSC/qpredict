import os
import logging
import csv
import os
import pandas as pd
import numpy as np

#neon
from neon.callbacks.callbacks import Callbacks
from neon.data.dataiterator import ArrayIterator
from neon.initializers import Xavier
from neon.layers import GeneralizedCost, Affine, Linear, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Adam, ExpSchedule
from neon.transforms import Rectlin#, MeanSquared
from neon.util.argparser import NeonArgparser
from sklearn import preprocessing
#from preprocess import feature_scaler #Thorsten's custom preprocessor
from cost import MeanSquaredLoss, MeanSquaredMetric, SmoothL1Loss, SmoothL1Metric

#plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

def y_fmt(x, y):
    return '{:2.2e}'.format(x).replace('e', '\,10^')
    

#start the main program
parser = NeonArgparser(__doc__)

args = parser.parse_args()
#irrespective of its value, set batchsize to one
args.batch_size=1

#do preprocessing
num_feat = 9
npzfile = np.load('jobwait_preproc.npz')
mean = npzfile['mean']
std = npzfile['std']
mean = np.reshape(mean, (1,mean.shape[0]))
std = np.reshape(std, (1,std.shape[0]))

# Reloading saved model
mlp=Model("jobwait_model.prm")
mlp.be.bsz=1

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
tmpmat[:,1:num_feat] -= mean
tmpmat[:,1:num_feat] /= std  

#create X and y-values
X_test=tmpmat[:,1:]
y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

# iterate over atomic batches:
errors=[]
for i in range(X_test.shape[0]/20):
    test_set = ArrayIterator(X=X_test[i:i+1,:], y=y_test[i:i+1,:], make_onehot=False)
    errors.append(mlp.eval(test_set, metric=SmoothL1Metric())[0])

#convert error-value from second to hours:
errors=[x/3600. for x in errors]

# obtain some statistics on the metric score
meanval=np.mean(errors)
medianval=np.median(errors)
central68=0.5*(np.percentile(errors, 84)-np.percentile(errors, 16))

#create histogram
Y,X=np.histogram(errors,bins=100)
width=0.25*(X[1]-X[0])
X=[0.5*(X[i]+X[i+1])-width/2 for i in range(len(X)-1)]
#select the maximal value in X which still has values
xmax=np.max(X)*1.1
#select the maximal value in Y times some factor
ymax=np.max(Y)*10

#plot histogram
fig = plt.figure()
ax = fig.add_subplot(111)

#ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: ('%.2f')%(x/3600)))
plt.yscale('log')
plt.xlim((0,xmax))
plt.ylim((0,ymax))
plt.xlabel('estimation error in hours')
plt.ylabel('log(#entries)')
plt.vlines(meanval, 0, ymax, colors='r', linestyles='solid', label='mean')
plt.bar(X,Y,width=width,label='data')
plt.savefig('stats/error_distribution.pdf')


