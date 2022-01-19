import os
from datetime import datetime

import numpy as np

from mace import generateSATExplanations
from mace.loadData import loadDataset
from mace.loadModel import loadModelForDataset
from mace.batchTest import runExperiments
from wachter.gen_counterfactual import Wachter
from roar.gen_counterfactual import ROAR

from utils import pad_ones, train_theta


model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', 'german')
theta, sigma = train_theta(pad_ones(X_train), y_train, 5)

wachter = Wachter(X_test, model_trained, linear=True)
roar = ROAR(X_test, model_trained.coef_.squeeze(), model_trained.intercept_, max_iter=50)
recourse = roar.fit_instance(X_test[0])
print(recourse)

print(model_trained.predict(recourse.reshape(1, -1)))
print(np.linalg.norm(X_test[0] - recourse))
