import os

import numpy as np
from sklearn.model_selection import train_test_split

import torch

from mace.loadModel import loadModelForDataset
from dirrac.classifier.mlp import mlp_classifier, MLP, train_mlp


def train_models_mlp(dataset_string, num_shuffle=10):
    print('Train model with dataset', dataset_string)

    path = os.path.join('result/models', dataset_string)
    if not os.path.exists(path):
        os.mkdir(path)

    # Load data
    model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', dataset_string)

    # Train model
    mlp = train_mlp(X_train, y_train)
    torch.save(mlp.state_dict(), os.path.join(path, 'model.pt'))

    for j in range(num_shuffle):
        X_train_shifted, X_test_shifted, y_train_shifted, y_test_shifted = train_test_split(X_shift, y_shift, test_size=0.1, random_state=j+1)
        clf_shifted = train_mlp(np.concatenate((X_train, X_test_shifted)), np.concatenate((y_train, y_test_shifted)))
        torch.save(clf_shifted.state_dict(), os.path.join(path, f'model_shift_{j}.pt'))


if __name__ == '__main__':
    train_models_mlp('german')
    train_models_mlp('sba')
    train_models_mlp('student')
