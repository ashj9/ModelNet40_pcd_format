import tensorflow as tf
import numpy as np
import os
import sys
import h5py

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
    
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)
    
current_data, current_label = loadDataFile('/home/ashj/Documents/ResearchDataset/ModelNets/modelnet40_ply_hdf5_2048/ply_data_train0.h5')
print(np.shape(current_data))
