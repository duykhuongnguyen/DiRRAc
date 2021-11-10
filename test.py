import numpy as np
from sklearn.model_selection import train_test_split

from dirrac.data.preprocess_data import GermanCredit, SBA, Student
from mace.loadModel import loadModelForDataset
from dirrac.classifier.logistic import logistic_classifier
from dirrac.gen_counterfactual import DRRA
from ar.gen_counterfactual import LinearAR
from roar.gen_counterfactual import ROAR
from mace.batchTest import runExperiments


model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', 'german')
X_recourse = X_test[model_trained.predict(X_test) == 0][:20]

roar = ROAR(X_recourse, model_trained, 0.1, sigma_max=0.2, alpha=0.1)
recourse = roar.fit_data(roar.data)
