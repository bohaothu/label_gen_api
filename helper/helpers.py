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

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)