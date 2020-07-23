# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import classic_nets_imagenet

from sklearn import svm


def get_transfer_values_classic_networks(dataset, network_name):

    # path to save the cache values
    file_path_cache_train = os.path.join(dataset.data_dir, network_name + '_' + dataset.name + '_train.pkl')
    file_path_cache_test = os.path.join(dataset.data_dir, network_name + '_' + dataset.name + '_test.pkl')


    print("\nTransferring Training Set...")

    if os.path.exists(file_path_cache_train):
        print("Training Set already exists on disk")
        with open(file_path_cache_train, "rb") as pick_file:
            transfer_values_train = pickle.load(pick_file)
    else:
        transfer_values_train = classic_nets_imagenet.classify_img(dataset.x_train, network_name)
        with open(file_path_cache_train, "wb") as pick_file:
            pickle.dump(transfer_values_train, pick_file)

    print("\nTransferring Test Set...")

    if os.path.exists(file_path_cache_test):
        print("Test Set already exists on disk")
        with open(file_path_cache_test, "rb") as pick_file:
            transfer_values_test = pickle.load(pick_file)
    else:
        transfer_values_test = classic_nets_imagenet.classify_img(dataset.x_test, network_name)
        with open(file_path_cache_test, "wb") as pick_file:
            pickle.dump(transfer_values_test, pick_file)

    return transfer_values_train, transfer_values_test


def transfer_values_svm_scores(train_x, train_y, test_x, test_y):
    clf = svm.SVC(probability=True)
    print("\nFitting SVM...")
    clf.fit(train_x, train_y)
    if len(test_x) != 0:
        print("\nEvaluating SVM...")
        test_scores = clf.predict_proba(test_x)
        print('Accuracy (SVM) = ', str(np.mean(np.argmax(test_scores, axis=1) == test_y)))
    else:
        test_scores = []
    train_scores = clf.predict_proba(train_x)
    return train_scores, test_scores

def svm_scores_exists(dataset, network_name="inception", alternative_data_dir="."):
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    return os.path.exists(svm_train_path) and os.path.exists(svm_test_path)

def get_svm_scores(transfer_values_train, y_train, transfer_values_test,
                   y_test, dataset, network_name="inception",
                   alternative_data_dir="."):
    
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    if not os.path.exists(svm_train_path) or not os.path.exists(svm_test_path):
        train_scores, test_scores = transfer_values_svm_scores(transfer_values_train, y_train, transfer_values_test, y_test)
        with open(svm_train_path, 'wb') as file_pi:
            pickle.dump(train_scores, file_pi)

        with open(svm_test_path, 'wb') as file_pi:
            pickle.dump(test_scores, file_pi)
    else:
        with open(svm_train_path, 'rb') as file_pi:
            train_scores = pickle.load(file_pi)

        with open(svm_test_path, 'rb') as file_pi:
            test_scores = pickle.load(file_pi)
    return train_scores, test_scores


def rank_data_according_to_score(train_scores, y_train, reverse=False, random=False):
    train_size, _ = train_scores.shape
    hardness_score = train_scores[list(range(train_size)), y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    if reverse:
        res = np.flip(res, 0)
    if random:
        np.random.shuffle(res)
    return res
