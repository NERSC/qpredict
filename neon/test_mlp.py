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
import matplotlib.patches as patches
import matplotlib.ticker as tick


#map of features to category-names
partitions=['debug', 'realtime', 'regular', 'regularx', 'shared']
qosclasses=['burstbuffer', 'debug', 'low', 'normal', 'premium', 'realtime', 'scavenger', 'serialize']

#start the main program
parser = NeonArgparser(__doc__)

args = parser.parse_args()
#irrespective of its value, set batchsize to one
args.batch_size=1

#do preprocessing
num_feat = 11
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

#extract feature matrix and label vector from test-dataframe
ncols=testdf.shape[1]
tmpmat=testdf.as_matrix()
tmpmat[:,1:num_feat] -= mean
tmpmat[:,1:num_feat] /= std  

#create X and y-values
X_test=tmpmat[:,1:]
y_test=np.reshape(tmpmat[:,0],(tmpmat[:,0].shape[0],1))

#do one forward pass to get estimates:
test_set = ArrayIterator(X=X_test, y=y_test, make_onehot=False)
y_pred=mlp.get_outputs(test_set).clip(min=10.)

# iterate over atomic batches:
errors=[]
for i in range(X_test.shape[0]):
    test_set = ArrayIterator(X=X_test[i:i+1,:], y=y_test[i:i+1,:], make_onehot=False)
	#append relative errors
    #errors.append(mlp.eval(test_set, metric=SmoothL1Metric())[0]/np.float(y_pred[i]))
    errors.append(mlp.eval(test_set, metric=SmoothL1Metric())[0])

##convert error-value from second to hours:
errors=[x/3600. for x in errors]

# obtain some statistics on the metric score
meanval=np.mean(errors)
medianval=np.median(errors)
central68=(np.percentile(errors, 16),np.percentile(errors, 84))
central95=(np.percentile(errors, 2.5),np.percentile(errors, 97.5))

print "Test error: mean = ", meanval,", median = ", medianval, ", 68% CL = [",central68[0],"; ",central68[1],"]"

#create histogram
Y,bins=np.histogram(errors,bins=50)
width=0.25*(bins[1]-bins[0])
X=[0.5*(bins[i]+bins[i+1])-width/2 for i in range(len(bins)-1)]
#select the maximal value in X which still has values
xmax=np.max(X)*1.1
#select the maximal value in Y times some factor
ymax=np.max(Y)*10

#plot histogram
fig = plt.figure()
ax = fig.add_subplot(111)

plt.yscale('log')
plt.xlim((0,xmax))
plt.ylim((0,ymax))
plt.xlabel('estimation error [h]')
plt.ylabel('log(#entries)')
bars=plt.bar(X,Y,width=width,label='data')
vlinemed=plt.vlines(medianval, 0, ymax, colors='r', linewidth=2, linestyles='solid', label='median')
rec=ax.add_patch(patches.Rectangle(
        (central95[0], 0),   # (x,y)
        central95[1]-central95[0],          # width
        ymax,          # height
        alpha=0.2,
        linewidth=0.5
    ))
blue_patch = patches.Patch(color='b',alpha=0.2 , label='95% CL')
plt.legend(handles=[bars, vlinemed, blue_patch])
plt.savefig('stats/error_distribution.pdf')

#merge the dataframe with the errors:
testdf['errors']=errors

#do plots depending on partition and qos-class:
fig, axvec= plt.subplots(figsize=(30,30), nrows=len(qosclasses), ncols=len(partitions))
for idx,pname in enumerate(partitions):
    for jdx,qname in enumerate(qosclasses):
        #get subfigure
        ax=axvec[jdx,idx]
        
        #get selection
        tmperr=list(testdf['errors'][ (testdf['qos_'+str(jdx)]==1) & (testdf['partition_'+str(idx)]==1) ])
        
        #plot histogram
        ax.set_title(pname+', '+qname)
        ax.set_yscale('log')
        ax.set_xlim((0,xmax))
        ax.set_ylim((0,ymax*10))
        ax.set_xlabel('estimation error [h]')
        ax.set_ylabel('log(#entries)')
        if tmperr:
            Y,Xtmp=np.histogram(tmperr,bins=bins)
            bars=ax.bar(X,Y,width=width,label='data')
        vlinemed=ax.vlines(medianval, 0, ymax, colors='r', linewidth=2, linestyles='solid', label='median')
        rec=ax.add_patch(patches.Rectangle(
                (central95[0], 0),   # (x,y)
                central95[1]-central95[0],          # width
                ymax,          # height
                alpha=0.2,
                linewidth=0.5
            ))
        blue_patch = patches.Patch(color='b',alpha=0.2 , label='95% CL')
        ax.legend(handles=[bars, vlinemed, blue_patch])

#save figure
plt.tight_layout()
fig.savefig('stats/error_distribution_split.pdf',bbox_inches='tight')
