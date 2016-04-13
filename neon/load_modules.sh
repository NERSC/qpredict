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

<<<<<<< HEAD
#module load python/2.7-anaconda
module load neon/1.3.0
=======
module load neon/1.3.0
>>>>>>> e77184a5600334203b3f851d326ef101ee414184
