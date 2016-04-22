import os
import sys

import numpy as np
import pandas as pd

def create_all_sets(hotdf, train_fraction, validation_fraction, test_fraction):
    numrows = hotdf.shape[0]
    rows    = np.arange(numrows)

    np.random.shuffle(rows)
    shuffledf = hotdf[hotdf.columns[1:]].iloc[rows].reset_index(drop=True)

    #training set
    train_stop = int(np.floor(train_fraction*numrows))
    shuffledf[:train_stop].to_csv('csv/cori_data_train.csv',header=True)
    #validation set
    validation_stop = int(np.floor(validation_fraction*numrows))+train_stop
    shuffledf[train_stop:validation_stop].to_csv('csv/cori_data_validate.csv',header=True)
    #test set:
    shuffledf[validation_stop:].to_csv('csv/cori_data_test.csv',header=True)
