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
import sklearn.cross_validation

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
elif a==100 or a==103:
    cluster=100
    total=2600
elif a==200:
    cluster=200
    total=1300


random_demand = np.zeros((cluster,total))
# Use all the SQL you like
for i in range(cluster):
        random_demand[i][:] = np.random.randint(1*(i+1), 15+1*(i+1), total)

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
                inputs += [np.concatenate((binary_day[i%7] , binary_department[i%24]) , axis = 1)]
                index += [[i, 1*(i+1), 15+1*(i+1)]]
        #fGroup.write(str(binary_day[1]) + ',' + str(binary_department[1]) + ',' + str(binary_month[1]) + ',' +  str(row)  + '\n')  
print "done" 

#inputs += [np.concatenate((binary_day[1] , binary_month[3]) , axis = 1)]
#fGroup.close()      

#-------------------------------------------------------------------------------------------------------------------------

# Split into train and test                                                                                                                     

X, Xt, y, yt, ind, indt = sklearn.cross_validation.train_test_split(inputs, suma, index, train_size=7500)

sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/uniform/IndexX-nw-10000-103-class', mdict={'IndexX': ind})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/uniform/IndexY-nw-10000-103-class', mdict={'IndexY': indt})

sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/uniform/TrainX-nw-10000-103-class', mdict={'trainX': X})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/uniform/TrainY-nw-10000-103-class', mdict={'trainY': y})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/uniform/TestX-nw-10000-103-class', mdict={'testX': Xt})
sio.savemat('/home/afo214/tensorflow/newsvendor/simulation/data/uniform/TestY-nw-10000-103-class', mdict={'testY': yt})

