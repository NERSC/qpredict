#!/bin/bash

#module load h5py
#module unload python
#module load python/2.7-anaconda
#module swap h5py h5py
#module load cray-hdf5
#module load mysql
#module load mysqlpython/1.2.5
#module load scikit-learn
#module load matplotlib

FOO=/global/homes/a/ankitb/qpredict/neon

module load python
source $FOO/.venv/bin/activate

export PATH=/usr/common/software/neon/1.3.0/bin:/usr/common/software/python/2.7.10/bin/:$PATH
export PYTHONPATH=/usr/common/software/neon/1.3.0/.venv/lib64/python2.7/site-packages:$PYTHONPATH

#module load neon/1.3.0
