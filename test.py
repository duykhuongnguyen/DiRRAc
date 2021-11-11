import numpy as np
from sklearn.model_selection import train_test_split

from dirrac.data.preprocess_data import GermanCredit, SBA, Student
from mace.loadModel import loadModelForDataset
from dirrac.classifier.logistic import logistic_classifier
from dirrac.classifier.mlp import mlp_classifier
from dirrac.gen_counterfactual import DRRA
from ar.gen_counterfactual import LinearAR
from roar.gen_counterfactual import ROAR
from mace.batchTest import runExperiments
from utils import train_non_linear, train_real_world_data

# validity = train_real_world_data('german', 1)
validity = train_non_linear('german', 10)
print(validity)
