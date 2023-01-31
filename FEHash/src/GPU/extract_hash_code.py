import tensorflow as tf
import numpy as np
import os
from numpy import random
from scipy import linalg, sparse
import numbers
import time

def read_emb(path):
    f = open(path, 'r')
    lines = f.readlines()
    emb_str = []
    for i in range(len(lines)):
        tmp = []
        tmp += lines[i].strip().split(" ")
        emb_str.append(tmp)
    f.close()

    emb = []
    for i in range(len(emb_str)):
        tmp = []
        for j in range(len(emb_str[i])):
            temp = float(emb_str[i][j])
            tmp.append(temp)

        emb.append(tmp)

    return emb

def read_state(path):
    f = open(path, 'r')
    lines = f.readlines()
    score = []
    for i in range(len(lines)):
        tmp = []
        tmp += lines[i].strip().split("\n")
        tmp = str(tmp[0])
        score.append(int(tmp))
    f.close()

    return score

def read_intercept(path):
    f = open(path, 'r')
    lines = f.readlines()
    score = []
    for i in range(len(lines)):
        tmp = []
        tmp += lines[i].strip().split("\n")
        tmp = str(tmp[0])
        score.append(float(tmp))
    f.close()

    return score

def safe_sparse_dot(a, b, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sparse.issparse(a) and sparse.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class projection_function():
    def __init__(self, gamma=1., n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, y=None):
        random_state = check_random_state(self.random_state)
        n_features = 512  # in Sec-FEHash, check this point, because we will reduce n_feature before train SVM

        self.random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(size=(n_features, self.n_components)))

        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)
        return self

def convert(gpu_pro):
    if(tf.sign(gpu_pro)==1):
        return 1
    else:
        return 0

def process(gpu_X, gpu_weight, gpu_offset, n_components, gpu_w, gpu_b):
    gpu_pro = tf.tensordot(gpu_X, gpu_weight, axes=1)
    gpu_pro = tf.math.add(gpu_offset, gpu_pro)
    gpu_pro = tf.math.cos(gpu_pro)
    gpu_pro = tf.math.multiply(gpu_pro, np.sqrt(2.) / np.sqrt(n_components))
    gpu_pro = tf.tensordot(gpu_pro, gpu_w, axes=1)
    gpu_pro = tf.math.add(gpu_pro, gpu_b)

    gpu_pro = tf.sign(gpu_pro)
    gpu_pro = tf.cast(gpu_pro, tf.int32)
    gpu_pro = tf.where(tf.equal(-1, gpu_pro), tf.zeros_like(gpu_pro), gpu_pro)

    return gpu_pro

##################################################################################################################################
path_emb_list = [r'final_tunable_embeddings_casia_1.txt',
                 r'final_tunable_embeddings_casia_2.txt',
                 r'final_tunable_embeddings_casia_3.txt',
                 r'final_tunable_embeddings_casia_4.txt',
                 r'final_tunable_embeddings_casia_5.txt']


path_param = r'casia_5120'

gamma = 2
compo = 5120 #adjust

####################################################################################################################################
for i in range(len(path_emb_list)):
    print("Embedding: ", path_emb_list[i])
    emb = read_emb(path_emb_list[i])
    tf_emb = tf.convert_to_tensor(emb, dtype=tf.float32)
    del emb


    path_coef = path_param + '\coef.txt'
    path_intercept = path_param + '\intercept.txt'
    path_state = path_param + '\state.txt'

    coef = read_emb(path_coef)
    intercept = read_intercept(path_intercept)
    state = read_state(path_state)

    results = []
    for k in range(len(state)):
        tmp = int(state[k])
        project_init = projection_function(gamma=gamma, n_components=compo, random_state=tmp)
        project_param = project_init.fit()

        tf_omega = tf.convert_to_tensor(project_param.random_weights_, dtype=tf.float32)
        tf_random_r = tf.convert_to_tensor(project_param.random_offset_, dtype=tf.float32)

        tf_intercept = tf.convert_to_tensor(intercept[k], dtype=tf.float32)
        tf_coef = tf.convert_to_tensor(coef[k], dtype=tf.float32)

        tf_result = process(tf_emb, tf_omega, tf_random_r, compo, tf_coef, tf_intercept)

        results.append(tf_result.numpy())

    del coef
    del intercept
    del state

    results = np.asarray(results).transpose()

    #write results
    print('Writing file: ', i)
    file_out = path_param + '\\final_tunable_results.txt'

    _hamming = open(file_out, 'a')
    for row in range(len(results)):
        for col in range(len(results[row])):
            print('%d' % results[row][col], end='', file=_hamming)
        print('\n', end='', file=_hamming)

    _hamming.close()

