import os
from typing import List
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, X, y, features):
        self.features = features
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.n_samples = self.X.shape[0]
        self.n_features = len(self.features)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

def get_train_data(data_path: str) -> Dataset:
    '''
    return torch Dataset for training, loading X_train and y_train with pickle from data_path.
    Dataset.X is a torch tensor with shape (n_samples, seq_len, n_features)
    Dataset.y is a torch tensor with shape (n_samples, 1)
    '''
    with open(os.path.join(data_path, 'X_train.pickle'), 'rb') as file:
        X_train = pkl.load(file)
    with open(os.path.join(data_path, 'y_train.pickle'), 'rb') as file:
        y_train = pkl.load(file)
    with open(os.path.join(data_path, 'features.pickle'), 'rb') as file:
        features = pkl.load(file)
    features = features.astype(str)
    train_data = dataset(X=X_train, y=y_train, features=features)
    return train_data

def get_test_data(data_path):
    '''
    return torch Dataset for testing, loading X_test and y_test with pickle from data_path.
    Dataset.X is a torch tensor with shape (n_samples, seq_len, n_features)
    Dataset.y is a torch tensor with shape (n_samples, 1)
    '''
    with open(os.path.join(data_path, 'X_test.pickle'), 'rb') as file:
        X_test = pkl.load(file)
    with open(os.path.join(data_path, 'y_test.pickle'), 'rb') as file:
        y_test = pkl.load(file)
    with open(os.path.join(data_path, 'features.pickle'), 'rb') as file:
        features = pkl.load(file)
    test_data = dataset(X=X_test, y=y_test, features=features)
    return test_data

def help():
    print('Available functions:')
    print('\tget_train_data')
    print('\tget_test_data')