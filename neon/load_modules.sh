#!/bin/bash

module swap cray-hdf5-parallel cray-hdf5
module swap h5py h5py
module swap python python/2.7-anaconda
module load mysql
module load mysqlpython/1.2.5
module load scikit-learn
module load matplotlib

module load neon/1.3.0_mkl