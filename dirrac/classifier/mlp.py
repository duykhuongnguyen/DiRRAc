import os
import pickle

from sklearn.neural_network import MLPClassifier


def mlp_classifier(X, y, save=False, save_dir=None):
    clf = MLPClassifier(hidden_layer_sizes=(64, 64, 64), random_state=1, max_iter=100)
    clf.fit(X, y)

    if save and save_dir:
        with open(save_dir, 'wb') as f:
            pickle.dump(clf, f)

    return clf
