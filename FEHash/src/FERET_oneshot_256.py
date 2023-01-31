import numpy as np
import math
from scipy import spatial
from sklearn import svm
from sklearn import metrics
from sklearn.kernel_approximation import RBFSampler
from time import time
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from itertools import combinations
import random

def read_ref(path):
    f = open(path, 'r')
    lines = f.readlines()
    ref = 0
    for i in range(1):
        tmp = []
        tmp += lines[i].strip().split("\n")
        tmp = str(tmp[0])
        ref = int(tmp)
    f.close()

    return ref

def create_combination_subset(n, r):
    return list(combinations(range(1, n+1), r))

def read_code(path):
    f = open(path, 'r')
    lines = f.readlines()
    codes = []
    for i in range(len(lines)):
        tmp = []
        tmp += lines[i].strip().split("\n")
        tmp = str(tmp[0])
        temp = list(tmp)
        codes.append(temp)
    f.close()

    return codes

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

############################################################
#DATASET
#file nay la file chinh dung de test state-of-the-art
path = r'C:\Users\ThaoDang\PycharmProjects\LSH-EC\bi_NIST_512_beacon\bi_NIST_1_108000.txt'
folder = r'D:\EC_LSH\FERET\FERET-different-align\FERET-eye-crop'
folder_aug = r'D:\EC_LSH\FERET\FERET-different-align\1_14_aug'
out_folder = r'D:\EC_LSH\FERET\FERET-different-align\save_param\FERET_256_round2'

codes = read_code(path)
user_list = []
emb = []
ref = []
ref_files = []
emb_aug = []
output_folders = []
folders = next(os.walk(folder))[1]
folders_aug = copy.deepcopy(folders)

for i in range(len(folders)):
    user_list.append(i)
    output_folders.append(out_folder + '\\' + folders[i])
    os.mkdir(output_folders[i]) #first time, next time remember to delete this line
    ref_files.append(folder + '\\' + folders[i] + '\\reference.txt')
    folders[i] = folder + '\\' + folders[i] + '\embeddings.txt'
    folders_aug[i] = folder_aug + '\\' + folders_aug[i] + '\embeddings.txt'

for file in ref_files:
    tmp = read_ref(file)
    ref.append(tmp)

for file in folders:
    tmp = read_emb(file)
    emb.append(tmp)

for file in folders_aug:
    tmp = read_emb(file)
    emb_aug.append(tmp)

###########################################################
#PARAMETER

n_enroll_list = [1]
n_padding_list = [21] #55
n_padding_choose = [10] #27

gamma = 2
compo_list = [5120]

############################################################
#RANDOM PADDING
for compo in compo_list:
    for n_enroll in n_enroll_list:
        print('Enroll: ', n_enroll)
        for n_padding in range(len(n_padding_list)):
            for user in range(len(emb)):
                set = create_combination_subset(n_padding_list[n_padding], n_padding_choose[n_padding])
                random.shuffle(set)

                print("User: ", user)
                genuine = user

                reference = ref[genuine]
                #genuine = 2
                impostor = copy.deepcopy(user_list)
                del impostor[genuine]

                padding = random.sample(impostor, k=n_padding_list[n_padding])

                tmp_quantum_key = codes[random.randint(0, 107999)]
                quantum_key = tmp_quantum_key[:-256]

                data_impostor = []
                for imp in impostor:
                    for copy_imp in range(len(emb[imp])):
                        data_impostor.append(emb[imp][copy_imp])

                ##################################################################################
                #WRITE FILE
                coef_file = []
                intercept_file = []
                state_file = []
                predict_file = []
                predict_impostor_file = []

                for ini in range(n_enroll, len(emb[user])):
                    predict_file.append('')

                for ini in range(len(data_impostor)):
                    predict_impostor_file.append('')

                ###################################################################################
                #TRAINING

                for i in range(len(quantum_key)): #planes = #of SVM
                    print("Plane: ", i)
                    label_train = []
                    data_train = []

                    data_test = []
                    #label_test = []

                    choose = set[i]

                    key = 0
                    opposite = 1

                    if(quantum_key[i] == '1'):
                        key = 1
                        opposite = 0

                    for j in range(n_enroll*14): #train
                    #for j in range(n_enroll*12):
                        data_train.append(emb_aug[genuine][j])
                        label_train.append(key)

                    for j in range(len(emb[genuine])):
                        if(j != reference):
                            data_test.append(emb[genuine][j])
                            #label_test.append(key)

                    for pad in padding:
                        for j in range(n_enroll*14):
                            data_train.append(emb_aug[pad][j])

                    for inte in range(1, len(padding)+1):
                        if(inte in choose):
                            for j in range(n_enroll*14):
                                label_train.append(key)
                        else:
                            for j in range(n_enroll*14):
                                label_train.append(opposite)

                    state = random.randint(1, 10000)

                    rbf_feature = RBFSampler(gamma=gamma, random_state=state, n_components=compo)
                    Z_train = rbf_feature.fit_transform(data_train)

                    clf = svm.SVC(kernel='linear')
                    clf.fit(Z_train, label_train)

                    state_file.append(state)
                    coef_file.append(clf.coef_[0])
                    intercept_file.append(clf.intercept_[0])

                    Z_test = rbf_feature.fit_transform(data_test)
                    predict = clf.predict(Z_test)
                    #print(metrics.accuracy_score(label_test, predict))

                    ######################################################################################################################
                    Z_impostor = rbf_feature.fit_transform(data_impostor)
                    predict_impostor = clf.predict(Z_impostor)

                    for write_predict in range(len(Z_test)):
                        predict_file[write_predict] = predict_file[write_predict] + str(predict[write_predict])

                    for write_predict in range(len(Z_impostor)):
                        predict_impostor_file[write_predict] = predict_impostor_file[write_predict] + str(predict_impostor[write_predict])
                    #########################################################################################################################

                _state = open(output_folders[genuine] + '\\' + str(gamma) + '_' + str(compo) + '_' + str(n_enroll) + '_' + str(n_padding_list[n_padding]) + '_state.txt', 'w')
                _coef = open(output_folders[genuine] + '\\' + str(gamma) + '_' + str(compo) + '_' + str(n_enroll) + '_' + str(n_padding_list[n_padding]) + '_coef.txt', 'w')
                _intercept = open(output_folders[genuine] + '\\' + str(gamma) + '_' + str(compo) + '_' + str(n_enroll) + '_' + str(n_padding_list[n_padding]) + '_intercept.txt', 'w')
                _quantum_key = open(output_folders[genuine] + '\\' + str(gamma) + '_' + str(compo) + '_' + str(n_enroll) + '_' + str(n_padding_list[n_padding]) + '_quantum_key.txt', 'w')
                _predict_key = open(output_folders[genuine] + '\\' + str(gamma) + '_' + str(compo) + '_' + str(n_enroll) + '_' + str(n_padding_list[n_padding]) + '_predict_user.txt', 'w')
                _predict_impostor_key = open(output_folders[genuine] + '\\' + str(gamma) + '_' + str(compo) + '_' + str(n_enroll) + '_' + str(n_padding_list[n_padding]) + '_predict_impostor.txt', 'w')

                for plane in range(len(quantum_key)):

                    print('%d' % state_file[plane], file=_state)
                    print('%f' % intercept_file[plane], file=_intercept)
                    print('%s' % quantum_key[plane], end='', file=_quantum_key)
                    for value in range(len(coef_file[plane])):
                        print('%f ' % coef_file[plane][value], end='', file=_coef)
                    print('\n', end='', file=_coef)

                for test_size in range(len(predict_file)):
                    print('%s' % predict_file[test_size], file=_predict_key)

                for impostor_size in range(len(predict_impostor_file)):
                    print('%s' % predict_impostor_file[impostor_size], file=_predict_impostor_key)

                _state.close()
                _coef.close()
                _intercept.close()
                _quantum_key.close()
                _predict_key.close()
                _predict_impostor_key.close()
