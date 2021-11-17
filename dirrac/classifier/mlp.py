import os
import pickle

from sklearn.neural_network import MLPClassifier


def mlp_classifier(X, y, random_state=None, save=False, save_dir=None):
    clf = MLPClassifier(hidden_layer_sizes=(64, 64, 64), random_state=random_state, max_iter=100, alpha=0.1)
    clf.fit(X, y)

    if save and save_dir:
        with open(save_dir, 'wb') as f:
            pickle.dump(clf, f)

    return clf
