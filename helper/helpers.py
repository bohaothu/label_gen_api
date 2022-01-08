import numpy as np
import pandas as pd

def gen_matrix_col(feature_n):
    #z = np.random.rand(1,feature_n)
    z = np.random.normal(12,30,size=(1,feature_n))
    z = z/np.sum(z)
    return np.transpose(z)

def gen_matrix(feature_n, label_n):
    return np.concatenate([ gen_matrix_col(feature_n) for x in range(label_n) ],axis=1)

def map_value(n):
    return n if n > 0.3 else 0

def gen_label(arr,gen_mat):
    return np.array([map_value(y) for y in np.dot(arr,gen_mat)])