import numpy as np
from sklearn.model_selection import train_test_split

import lime
import lime.lime_tabular


np.random.seed(0)


class LocalApprox(object):

    def __init__(self, X_train, predict_fn):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(X_train, class_names=['0', '1'], discretize_continuous=True, sample_around_instance=True, random_state=0)
        self.predict_fn = predict_fn

    def extract_weights(self, x_0, shift=0.1):
        exp = self.explainer.explain_instance(x_0, self.predict_fn, top_labels=1, num_features=100, num_samples=5000)
        coefs = exp.local_exp[0]
        intercept = exp.intercept[0]
        coefs = sorted(coefs, key=lambda x: x[0])

        w = np.array([e[1] for e in coefs])
        b = -shift - np.dot(w, x_0)

        return w, np.array(b).reshape(1,)
