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
from lime_explainer.local_approx import LocalApprox


def pad_ones(data, ax=1):
    if ax == 1:
        pad_data = np.concatenate((data, np.ones(len(data)).reshape(-1, 1)), axis=ax)
    else:
        pad_data = np.concatenate((data, np.ones(1)))
    return pad_data


def train_theta(X, y, num_shuffle):
    dim = X.shape[1]
    all_coef = np.zeros((num_shuffle, dim))
    for i in range(num_shuffle):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(i + 1) * 5)
        coef = logistic_classifier(X_train, y_train)[1].T
        all_coef[i] = np.squeeze(coef)

    theta = np.zeros((1, dim))
    sigma = np.zeros((1, dim, dim))
    theta[0], sigma[0] = np.mean(all_coef, axis=0), np.cov(all_coef.T)

    return theta, sigma


def cal_cost(samples, counterfactual_samples, type_cost='l1'):
    l1 = np.mean(np.linalg.norm((samples - counterfactual_samples), ord=1, axis=1))
    s1 = np.std(np.linalg.norm((samples - counterfactual_samples), ord=1, axis=1))
    l2 = np.mean(np.linalg.norm((samples - counterfactual_samples), axis=1))
    s2 = np.std(np.linalg.norm((samples - counterfactual_samples), axis=1))
    return [l1, s1] if type_cost == 'l1' else [l2, s2]


def cal_validity(pred):
    return sum(pred == 1) / len(pred)


def train_real_world_data(dataset_string, num_samples, real_data=True, padding=True):
    # Load data
    model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', dataset_string)
    
    X_recourse = X_test[model_trained.predict(X_test) == 0][:num_samples]

    # Initialize modules
    beta = 0
    delta = 0.2
    k = 1
    p = np.array([1])
    rho = np.array([0])
    lmbda = 0.7
    zeta = 1
    theta, sigma = train_theta(pad_ones(np.concatenate((X_train, X_test))), np.concatenate((y_train, y_test)), 10)
    drra_module = DRRA(delta, k, X_train.shape[1] + 1, p, theta, sigma * (1 + beta), rho, lmbda, zeta, dist_type='l1', real_data=real_data, padding=padding)

    ar_module = LinearAR(X_train, theta[:, :-1], theta[0][-1])
    roar = ROAR(X_recourse, model_trained.coef_.squeeze(), model_trained.intercept_, 0.1, sigma_max=0.2, alpha=1e-2, dist_type='l1')

    validity = {'AR': [0, 0, 0, 0, 0, 0], 'MACE': [0, 0, 0, 0, 0, 0], 'ROAR': [0, 0, 0, 0, 0, 0], 'DiRRAc-NM': [0, 0, 0, 0, 0, 0], 'DiRRAc-GM': [0, 0, 0, 0, 0, 0]}

    # Generate counterfactual
    print("Generate counterfactual for DiDRAc-NM")
    counterfactual_drra_nm = drra_module.fit_data(pad_ones(X_recourse))
    print("Generate counterfactual for DiDRAc-GM")
    counterfactual_drra_gm = drra_module.fit_data(pad_ones(X_recourse), model='gm')
    print("Generate counterfactual for AR")
    counterfactual_ar = ar_module.fit_data(X_recourse)
    print("Generate counterfactual for MACE")
    counterfactual_mace = runExperiments([dataset_string], ['lr'], ['one_norm'], ['MACE_eps_1e-5'], 0, len(X_recourse), 'neg_only', '0', theta[:, :-1], theta[:, -1])
    print("Generate counterfactual for ROAR")
    counterfactual_roar = roar.fit_data(roar.data)

    drra_nm, drra_gm, ar, mace, roar = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
    # Train model with shifted data
    for i in range(100):
        X_train_shifted, X_test_shifted, y_train_shifted, y_test_shifted = train_test_split(X_shift, y_shift, test_size=0.2, random_state=i+1)
        clf_shifted = logistic_classifier(X_train_shifted, y_train_shifted, intercept=True)[0]

        drra_nm[i] = cal_validity(clf_shifted.predict(counterfactual_drra_nm[:, :-1]))
        drra_gm[i] = cal_validity(clf_shifted.predict(counterfactual_drra_gm[:, :-1]))
        ar[i] = cal_validity(clf_shifted.predict(counterfactual_ar))
        mace[i] = cal_validity(clf_shifted.predict(counterfactual_mace))
        roar[i] = cal_validity(clf_shifted.predict(counterfactual_roar[:, :-1]))

    validity['AR'] = [np.mean(ar), np.std(ar)] + cal_cost(counterfactual_ar, X_recourse) + cal_cost(counterfactual_ar, X_recourse, 'l2')
    validity['MACE'] = [np.mean(mace), np.std(mace)] + cal_cost(counterfactual_mace, X_recourse) + cal_cost(counterfactual_mace, X_recourse, 'l2')
    validity['DiRRAc-NM'] = [np.mean(drra_nm), np.std(drra_nm)] + cal_cost(counterfactual_drra_nm[:, :-1], X_recourse) + cal_cost(counterfactual_drra_nm[:, :-1], X_recourse, 'l2')
    validity['DiRRAc-GM'] = [np.mean(drra_gm), np.std(drra_gm)] + cal_cost(counterfactual_drra_gm[:, :-1], X_recourse) + cal_cost(counterfactual_drra_gm[:, :-1], X_recourse, 'l2')
    validity['ROAR'] = [np.mean(roar), np.std(roar)] + cal_cost(counterfactual_roar[:, :-1], X_recourse) + cal_cost(counterfactual_roar[:, :-1], X_recourse, 'l2')

    return validity


def train_non_linear(dataset_string, num_samples, real_data=True, padding=True):
    # Load data
    model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', dataset_string)
    mlp = mlp_classifier(X_train, y_train)
    X_recourse = X_test[mlp.predict(X_test) == 0][:num_samples]

    # Local approximation
    local_approx = LocalApprox(X_train, mlp.predict_proba)
    all_coef = np.zeros((100, X_train.shape[1] + 1))
    for i in range(10):
        coef, intercept = local_approx.extract_weights(X_recourse[1])
        all_coef[i] = np.concatenate((coef, intercept))
    theta = np.zeros((1, X_train.shape[1] + 1))
    sigma = np.zeros((1, X_train.shape[1] + 1, X_train.shape[1] + 1))
    theta[0], sigma[0] = np.mean(all_coef, axis=0), np.cov(all_coef.T)

    # theta, sigma = np.concatenate((coef, intercept)).reshape(1, -1), np.expand_dims(0.01 * np.identity(X_train.shape[1] + 1), axis=0)

    # Initialize modules
    beta = 0
    delta = 0.2
    k = 1
    p = np.array([1])
    rho = np.array([0])
    lmbda = 0.7
    zeta = 1
    # theta, sigma = train_theta(pad_ones(np.concatenate((X_train, X_test))), np.concatenate((y_train, y_test)), 10)
    drra_module = DRRA(delta, k, X_train.shape[1] + 1, p, theta, sigma * (1 + beta), rho, lmbda, zeta, dist_type='l1', real_data=real_data, padding=padding)

    ar_module = LinearAR(X_train, theta[:, :-1], theta[0][-1])
    roar = ROAR(X_recourse, coef.squeeze(), intercept, 0.1, sigma_max=0.2, alpha=1e-2, dist_type='l1')

    validity = {'AR': [0, 0, 0, 0, 0, 0], 'MACE': [0, 0, 0, 0, 0, 0], 'ROAR': [0, 0, 0, 0, 0, 0], 'DiRRAc-NM': [0, 0, 0, 0, 0, 0], 'DiRRAc-GM': [0, 0, 0, 0, 0, 0]}

    # Generate counterfactual
    print("Generate counterfactual for DiDRAc-NM")
    counterfactual_drra_nm = drra_module.fit_instance(pad_ones(X_recourse[1], ax=0))
    print(np.dot(theta, counterfactual_drra_nm) + intercept, mlp.predict_proba(counterfactual_drra_nm[:-1].reshape(1, -1)))
    print("Generate counterfactual for DiDRAc-GM")
    counterfactual_drra_gm = drra_module.fit_instance(pad_ones(X_recourse[1], ax=0), model='gm')
    print("Generate counterfactual for AR")
    counterfactual_ar = ar_module.fit_instance(X_recourse[1])
    print("Generate counterfactual for MACE")
    counterfactual_mace = runExperiments([dataset_string], ['lr'], ['one_norm'], ['MACE_eps_1e-5'], 0, 1, 'neg_only', '0', theta[:, :-1], theta[:, -1])
    print("Generate counterfactual for ROAR")
    counterfactual_roar = roar.fit_instance(roar.data[1])

    drra_nm, drra_gm, ar, mace, roar = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
    # Train model with shifted data
    for i in range(100):
        X_train_shifted, X_test_shifted, y_train_shifted, y_test_shifted = train_test_split(X_shift, y_shift, test_size=0.2, random_state=i+1)
        clf_shifted = mlp_classifier(X_train_shifted, y_train_shifted)

        drra_nm[i] = clf_shifted.predict(counterfactual_drra_nm[:-1].reshape(1, -1))
        drra_gm[i] = clf_shifted.predict(counterfactual_drra_gm[:-1]. reshape(1, -1))
        ar[i] = clf_shifted.predict(counterfactual_ar.reshape(1, -1))
        mace[i] = clf_shifted.predict(counterfactual_mace.reshape(1, -1))
        roar[i] = clf_shifted.predict(counterfactual_roar[:-1].reshape(1, -1))

    validity['AR'] = [np.mean(ar), np.std(ar)] + [0] * 4# cal_cost(counterfactual_ar, X_recourse) + cal_cost(counterfactual_ar, X_recourse, 'l2')
    validity['MACE'] = [np.mean(mace), np.std(mace)] + [0] * 4 #cal_cost(counterfactual_mace, X_recourse) + cal_cost(counterfactual_mace, X_recourse, 'l2')
    validity['DiRRAc-NM'] = [np.mean(drra_nm), np.std(drra_nm)] + [0] * 4 # + cal_cost(counterfactual_drra_nm[:, :-1], X_recourse) + cal_cost(counterfactual_drra_nm[:, :-1], X_recourse, 'l2')
    validity['DiRRAc-GM'] = [np.mean(drra_gm), np.std(drra_gm)] + [0] * 4 # cal_cost(counterfactual_drra_gm[:, :-1], X_recourse) + cal_cost(counterfactual_drra_gm[:, :-1], X_recourse, 'l2')
    validity['ROAR'] = [np.mean(roar), np.std(roar)] + [0] * 4 #cal_cost(counterfactual_roar[:, :-1], X_recourse) + cal_cost(counterfactual_roar[:, :-1], X_recourse, 'l2')
    return validity
