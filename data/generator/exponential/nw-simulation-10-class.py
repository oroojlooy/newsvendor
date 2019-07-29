import scipy.io as sio
import numpy as np
import os

import sys
import h5py
import shutil
import tempfile

import sklearn
import sklearn.datasets
import sklearn.linear_model

import scipy.stats as stso
np.random.seed(seed=4)

import sys
a = int(sys.argv[0].split('-')[2])

if a == 1:
    cluster=1
    total=260000
elif a==10:
    cluster=10
    total=26000
elif a==100:
    cluster=100
    total=2600
elif a==200:
    cluster=200
    total=1300


random_demand = np.zeros((cluster,total))
# Use all the SQL you like
for i in range(cluster):
        random_demand[i][:] = np.random.exponential(5 + 2*(i+1), total)

binary_day = []
binary_month = []
binary_department = []

A = np.identity(7)
for i in range(7):
        binary_day += [A[i : i + 1]]

A = np.identity(10)
for i in range(12):
        binary_month += [A[i : i + 1]]
        
A = np.identity(24)
for i in range(24):
        binary_department += [A[i : i + 1]]

inputs = []
suma = []
index = []

for i in range(cluster):
        for demand in random_demand[i][:]:
                suma += [[round(demand)]]
                #inputs += [np.concatenate((binary_day[np.random.randint(1,3)] , binary_department[np.random.randint(1,6)]) , axis = 1)]
                inputs += [np.concatenate((binary_day[i%2 + 1] , binary_department[i%5 + 1]) , axis = 1)]
                index += [[i,5 + 2*(i+1)]]
        #fGroup.write(str(binary_day[1]) + ',' + str(binary_department[1]) + ',' + str(binary_month[1]) + ',' +  str(row)  + '\n')  
print "done" 

#inputs += [np.concatenate((binary_day[1] , binary_month[3]) , axis = 1)]
#fGroup.close()      

#-------------------------------------------------------------------------------------------------------------------------

# Split into train and test                                                                                                                     

X, Xt, y, yt, ind, indt = sklearn.cross_validation.train_test_split(inputs, suma, index, train_size=7500)

sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/exponential/IndexX-nw-10000-10-class', mdict={'IndexX': ind})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/exponential/IndexY-nw-10000-10-class', mdict={'IndexY': indt})

sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/exponential/TrainX-nw-10000-10-class', mdict={'trainX': X})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/exponential/TrainY-nw-10000-10-class', mdict={'trainY': y})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/exponential/TestX-nw-10000-10-class', mdict={'testX': Xt})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/exponential/TestY-nw-10000-10-class', mdict={'testY': yt})

print "done" 

Trainh5 = 'Train-nw-10000-10-class.h5'
Traintxt = 'Train-nw-10000-10-class.txt'
Testh5 = 'Test-nw-10000-10-class.h5'
Testtxt = 'Test-nw-10000-10-class.txt'

# Write out the data to HDF5 files in a temp directory.                                                                                         # This file is assumed to be caffe_root/examples/hdf5_classification.ipynb                                                                       
dirname = os.path.abspath('/home/afo214/tensorflow/newsvendor/simulation/data/hdf5')
if not os.path.exists(dirname):
    os.makedirs(dirname)

train_filename = os.path.join(dirname, Trainh5)
test_filename = os.path.join(dirname, Testh5)

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.                                                                     
# To show this off, we'll list the same data file twice.                                                                                         
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)
with open(os.path.join(dirname, Traintxt), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')

# HDF5 is pretty efficient, but can be further compressed. 
                                                                                    
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    train_filename = os.path.join(dirname, Trainh5)
    test_filename = os.path.join(dirname, Testh5)

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.                                                                     
# To show this off, we'll list the same data file twice.                                                                                         
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)
with open(os.path.join(dirname,Traintxt), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')

# HDF5 is pretty efficient, but can be further compressed.                                                                                       
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, Testtxt), 'w') as f:
    f.write(test_filename + '\n')

