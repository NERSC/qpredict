#!/bin/bash

module load h5py
module unload python
module load python/2.7-anaconda
module swap h5py h5py
module load cray-hdf5
module load mysql
module load mysqlpython/1.2.5
module load scikit-learn
module load matplotlib

module load neon/1.1.4
