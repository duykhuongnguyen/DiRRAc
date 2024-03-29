import time
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

import torch

from dirrac.data.preprocess_data import GermanCredit, SBA, Student
from mace.loadModel import loadModelForDataset
from dirrac.classifier.logistic import logistic_classifier
from dirrac.classifier.mlp import mlp_classifier, MLP, train_mlp
from dirrac.gen_counterfactual import DRRA
from ar.gen_counterfactual import LinearAR
from roar.gen_counterfactual import ROAR
from wachter.gen_counterfactual import Wachter
from lime_explainer.local_approx import LocalApprox
from mace.batchTest import runExperiments


np.random.seed(0)


def pad_ones(data, ax=1):
    if ax == 1:
        pad_data = np.concatenate((data, np.ones(len(data)).reshape(-1, 1)), axis=ax)
    else:
        pad_data = np.concatenate((data, np.ones(1)))
    return pad_data


def train_theta(X, y, num_shuffle, n_components=1):
    dim = X.shape[1]
    all_coef = np.zeros((num_shuffle, dim))
    for i in range(num_shuffle):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(i + 1) * 5)
        coef = logistic_classifier(X_train, y_train)[1].T
        all_coef[i] = np.squeeze(coef)

    if n_components == 1:
        theta = np.zeros((1, dim))
        sigma = np.zeros((1, dim, dim))
        theta[0], sigma[0] = np.mean(all_coef, axis=0), np.cov(all_coef.T)

        return theta, sigma, np.array([1])

    gm = GaussianMixture(n_components=n_components, random_state=0).fit(all_coef)
    
    return gm.means_, gm.covariances_, gm.weights_


def cal_cost(samples, counterfactual_samples, type_cost='l1'):
    l1 = np.mean(np.linalg.norm((samples - counterfactual_samples), ord=1, axis=1))
    s1 = np.std(np.linalg.norm((samples - counterfactual_samples), ord=1, axis=1))
    l2 = np.mean(np.linalg.norm((samples - counterfactual_samples), axis=1))
    s2 = np.std(np.linalg.norm((samples - counterfactual_samples), axis=1))
    return [l1, s1] if type_cost == 'l1' else [l2, s2]


def cal_validity(pred):
    return sum(pred == 1) / len(pred)


def train_real_world_data(dataset_string, num_samples, real_data=True, padding=True, sigma_identity=False, actionable=False, n_components=1):
    # Load data
    model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', dataset_string)
    X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
    X_recourse = X_test[model_trained.predict(X_test) == 0][:num_samples]

    # Initialize modules
    beta = 0
    delta = 3.0 # if dataset_string == 'german' else 0.6
    # delta = 0.6
    k = n_components
    rho = np.array([0 for i in range(k)])
    lmbda = 0.7
    zeta = 1
    theta, sigma, p = train_theta(pad_ones(X_train), y_train, 100, k)

    # Run experiments with sigma identity
    if sigma_identity:
        theta[0, :] = np.concatenate([model_trained.coef_.squeeze(), model_trained.intercept_])
        sigma[0, :, :] = 0.1 * np.identity(sigma.shape[1])
        delta = 3.0

    # Cat indices
    cat_indices = {'german': [[4, 5, 6, 7]], 'sba': [[8, 9, 10], [11, 12], [13, 14], [15, 16], [17, 18]], 'student': [[4, 5, 6, 7], [8, 9], [10, 11], [12, 13], [14, 15, 16, 17, 18]]}
    num_discrete = {'german': 4, 'sba': 8, 'student': 4}

    # Initialize modules
    ar_module = LinearAR(X_train, model_trained.coef_.squeeze(), model_trained.intercept_, encoding_constraints=True, dis_indices=num_discrete[dataset_string])
    roar = ROAR(X_recourse, model_trained.coef_.squeeze(), model_trained.intercept_, lmbda=1e-3, alpha=0.5, max_iter=100, encoding_constraints=True, cat_indices=cat_indices[dataset_string])
    wachter = Wachter(X_recourse, model_trained, decision_threshold=0.5, max_iter=1000, linear=True, encoding_constraints=True, cat_indices=cat_indices[dataset_string])

    # Add actionability constraints
    if actionable:
        immutable_d = {'german': [2], 'sba': [X_train.shape[1] - 2, X_train.shape[1] - 1, X_train.shape[1] - 7, X_train.shape[1] - 8, X_train.shape[1] - 9], 'student': [8, 9, 10, 11]}
        non_icr_d = {'german': [0, 3], 'sba': [4], 'student': [0, 1, 2, 3]}
        drra_module = DRRA(delta, k, X_train.shape[1] + 1, p, theta, sigma * (1 + beta), rho, lmbda, zeta, dist_type='l1', real_data=real_data, num_discrete=num_discrete[dataset_string], padding=padding, immutable_l=immutable_d[dataset_string], non_icr_l=non_icr_d[dataset_string], cat_indices=cat_indices[dataset_string])
        ar_module = LinearAR(X_train, model_trained.coef_.squeeze(), model_trained.intercept_, encoding_constraints=True, dis_indices=num_discrete[dataset_string], immutable_l=immutable_d[dataset_string], non_icr_l=non_icr_d[dataset_string])
    else:
        drra_module = DRRA(delta, k, X_train.shape[1] + 1, p, theta, sigma * (1 + beta), rho, lmbda, zeta, dist_type='l1', real_data=real_data, padding=padding, cat_indices=cat_indices[dataset_string])

    validity = {'AR': [0, 0, 0, 0, 0, 0, 0, 0], 'Wachter': [0, 0, 0, 0, 0, 0, 0, 0], 'ROAR': [0, 0, 0, 0, 0, 0, 0, 0], 'DiRRAc': [0, 0, 0, 0, 0, 0, 0, 0], 'Gaussian DiRRAc': [0, 0, 0, 0, 0, 0, 0, 0]}

    # Generate counterfactual
    print("Generate counterfactual for DiDRAc-NM")
    t = time.time()
    counterfactual_drra_nm = drra_module.fit_data(pad_ones(X_recourse))
    print("Generate counterfactual for DiDRAc-GM")
    t = time.time()
    counterfactual_drra_gm = drra_module.fit_data(pad_ones(X_recourse), model='gm')
    print("Generate counterfactual for AR")
    t = time.time()
    counterfactual_ar = ar_module.fit_data(X_recourse)
    print("Generate counterfactual for ROAR")
    t = time.time()
    counterfactual_roar = roar.fit_data(roar.data)
    print("Generate counterfactual for Wachter")
    t = time.time()
    counterfactual_wachter = wachter.fit_data(wachter.data)

    drra_nm_m1, drra_gm_m1, ar_m1, ar_hat_m1,  roar_m1, roar_hat_m1, wachter_m1, wachter_hat_m1, mint_m1 = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
    
    # Evaluate with with original classifier
    drra_nm_m1 = model_trained.predict(counterfactual_drra_nm[:, :-1])
    drra_gm_m1 = model_trained.predict(counterfactual_drra_gm[:, :-1])
    ar_m1 = model_trained.predict(counterfactual_ar)
    roar_m1 = model_trained.predict(counterfactual_roar)
    wachter_m1 = model_trained.predict(counterfactual_wachter)

    drra_nm, drra_gm, ar, ar_hat, roar, roar_hat, wachter, wachter_hat, mint = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
    # Train model with shifted data
    for i in range(100):
        X_train_shifted, X_test_shifted, y_train_shifted, y_test_shifted = train_test_split(X_shift, y_shift, test_size=0.2, random_state=i+1)
        clf_shifted = logistic_classifier(X_train_shifted, y_train_shifted, intercept=True)[0]

        drra_nm[i] = cal_validity(clf_shifted.predict(counterfactual_drra_nm[:, :-1]))
        drra_gm[i] = cal_validity(clf_shifted.predict(counterfactual_drra_gm[:, :-1]))
        ar[i] = cal_validity(clf_shifted.predict(counterfactual_ar))
        roar[i] = cal_validity(clf_shifted.predict(counterfactual_roar))
        wachter[i] = cal_validity(clf_shifted.predict(counterfactual_wachter))

    validity['AR'] = [np.mean(ar_m1), np.std(ar_m1)] + [np.mean(ar), np.std(ar)] + cal_cost(counterfactual_ar, X_recourse) + cal_cost(counterfactual_ar, X_recourse, 'l2')
    validity['DiRRAc'] = [np.mean(drra_nm_m1), np.std(drra_nm_m1)] + [np.mean(drra_nm), np.std(drra_nm)] + cal_cost(counterfactual_drra_nm[:, :-1], X_recourse) + cal_cost(counterfactual_drra_nm[:, :-1], X_recourse, 'l2')
    validity['Gaussian DiRRAc'] = [np.mean(drra_gm_m1), np.std(drra_gm_m1)] + [np.mean(drra_gm), np.std(drra_gm)] + cal_cost(counterfactual_drra_gm[:, :-1], X_recourse) + cal_cost(counterfactual_drra_gm[:, :-1], X_recourse, 'l2')
    validity['ROAR'] = [np.mean(roar_m1), np.std(roar_m1)] + [np.mean(roar), np.std(roar)] + cal_cost(counterfactual_roar, X_recourse) + cal_cost(counterfactual_roar, X_recourse, 'l2')
    validity['Wachter'] = [np.mean(wachter_m1), np.std(wachter_m1)] + [np.mean(wachter), np.std(wachter)] + cal_cost(counterfactual_wachter, X_recourse) + cal_cost(counterfactual_wachter, X_recourse, 'l2')

    return validity


def train_non_linear(dataset_string, num_samples, real_data=True, padding=True, num_shuffle=2):
    # Load data
    model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', dataset_string)
    X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
    
    # Load model
    mlp = MLP(X.shape[1])
    mlp.load_state_dict(torch.load(f'result/models_0/{dataset_string}/model.pt'))

    model_shifted = []
    for i in range(10):
        clf_shifted = MLP(X.shape[1])
        clf_shifted.load_state_dict(torch.load(f'result/models_0/{dataset_string}/model_shift_{i}.pt'))
        model_shifted.append(clf_shifted)
    
    X_recourse = X_test[mlp.predict(X_test) == 0][:num_samples]

    # Cat indices
    cat_indices = {'german': [[4, 5, 6, 7]], 'sba': [[8, 9, 10], [11, 12], [13, 14], [15, 16], [17, 18]], 'student': [[4, 5, 6, 7], [8, 9], [10, 11], [12, 13], [14, 15, 16, 17, 18]]}
    num_discrete = {'german': 4, 'sba': 8, 'student': 4}

    # Initialize modules
    beta = 0
    delta = 0.8
    k = 1
    p = np.array([1])
    rho = np.array([0])
    lmbda = 0.7
    zeta = 1
    num_discrete = {'german': 4, 'sba': 8, 'student': 4}

    validity = {'AR': [0, 0, 0, 0, 0, 0, 0, 0], 'Wachter': [0, 0, 0, 0, 0, 0, 0, 0], 'ROAR': [0, 0, 0, 0, 0, 0, 0, 0], 'DiRRAc': [0, 0, 0, 0, 0, 0, 0, 0], 'Gaussian DiRRAc': [0, 0, 0, 0, 0, 0, 0, 0]}
    drra_nm_m1, drra_gm_m1, ar_m1, roar_m1, wachter_m1 = [], [], [], [], []
    drra_nm_m2, drra_gm_m2, ar_m2, roar_m2, wachter_m2 = [], [], [], [], []
    counterfactual_drra_nm_l, counterfactual_drra_gm_l, counterfactual_ar_l, counterfactual_roar_l, counterfactual_wachter_l = [], [], [], [], []
    X_recourse_ = []

    shift_bound = {'german': 0.1, 'sba': 0.1, 'student': 0.1}
    for i in range(len(X_recourse)):
        # Local approximation
        local_approx = LocalApprox(X_train, mlp.predict_proba)
        all_coef = np.zeros((10, X_train.shape[1] + 1))
        for j in range(10):
            coef, intercept = local_approx.extract_weights(X_recourse[i], shift=shift_bound[dataset_string])
            all_coef[j] = np.concatenate((coef, intercept))
        theta = np.zeros((1, X_train.shape[1] + 1))
        sigma = np.zeros((1, X_train.shape[1] + 1, X_train.shape[1] + 1))
        theta[0], sigma[0] = np.mean(all_coef, axis=0), np.cov(all_coef.T)

        # Initialize modules
        drra_module = DRRA(delta, k, X_train.shape[1] + 1, p, theta, sigma * (1 + beta), rho, lmbda, zeta, dist_type='l1', real_data=real_data, padding=padding, cat_indices=cat_indices[dataset_string])
        ar_module = LinearAR(X_train, theta[:, :-1], theta[0][-1], encoding_constraints=True, dis_indices=num_discrete[dataset_string])
        roar = ROAR(X_recourse, theta[:, :-1].squeeze(), torch.tensor([theta[0][-1]]), lmbda=1e-3, alpha=0.2, max_iter=30, encoding_constraints=True, cat_indices=cat_indices[dataset_string])
        wachter = Wachter(X_recourse, mlp, max_iter=1000, linear=False, encoding_constraints=True, cat_indices=cat_indices[dataset_string])

        # Generate counterfactual
        counterfactual_ar = ar_module.fit_instance(X_recourse[i])
        if np.linalg.norm(counterfactual_ar) == 0:
            continue

        counterfactual_drra_nm = drra_module.fit_instance(pad_ones(X_recourse[i], ax=0))
        X_recourse_.append(X_recourse[i])
        print("Generate counterfactual for DiDRAc-NM")
        counterfactual_drra_nm_l.append(counterfactual_drra_nm)
        print("Generate counterfactual for DiDRAc-GM")
        counterfactual_drra_gm = drra_module.fit_instance(pad_ones(X_recourse[i], ax=0), model='gm')
        counterfactual_drra_gm_l.append(counterfactual_drra_gm)
        print("Generate counterfactual for AR")
        counterfactual_ar_l.append(np.squeeze(counterfactual_ar))
        print("Generate counterfactual for ROAR")
        counterfactual_roar = roar.fit_instance(roar.data[i])
        counterfactual_roar_l.append(counterfactual_roar)
        print("Generate counterfactual for Wachter")
        counterfactual_wachter = wachter.fit_instance(wachter.data[i])
        counterfactual_wachter_l.append(counterfactual_wachter)

        drra_nm_, drra_gm_, ar_, roar_, wachter_ = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        # Train model with data
        for j in range(1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i+1)
            clf = mlp_classifier(X_train, y_train, random_state=j+1)

            drra_nm_[j] = mlp.predict(counterfactual_drra_nm[:-1].reshape(1, -1))
            drra_gm_[j] = mlp.predict(counterfactual_drra_gm[:-1]. reshape(1, -1))
            try:
                ar_[j] = mlp.predict(counterfactual_ar.reshape(1, -1))
            except:
                ar_[j] = 0
            roar_[j] = mlp.predict(counterfactual_roar.reshape(1, -1))
            wachter_[j] = mlp.predict(counterfactual_wachter.reshape(1, -1))

        drra_nm_m1.append(np.mean(drra_nm_))
        drra_gm_m1.append(np.mean(drra_gm_))
        ar_m1.append(np.mean(ar_))
        roar_m1.append(np.mean(roar_))
        wachter_m1.append(np.mean(wachter_))

        drra_nm_, drra_gm_, ar_, roar_, wachter_ = np.zeros(num_shuffle), np.zeros(num_shuffle), np.zeros(num_shuffle), np.zeros(num_shuffle), np.zeros(num_shuffle)
        # Train model with shifted data
        for j in range(num_shuffle):
            X_train_shifted, X_test_shifted, y_train_shifted, y_test_shifted = train_test_split(X_shift, y_shift, test_size=0.05, random_state=i+1)
            clf_shifted = model_shifted[j]

            drra_nm_[j] = clf_shifted.predict(counterfactual_drra_nm[:-1].reshape(1, -1))
            drra_gm_[j] = clf_shifted.predict(counterfactual_drra_gm[:-1]. reshape(1, -1))
            try:
                ar_[j] = clf_shifted.predict(counterfactual_ar.reshape(1, -1))
            except:
                ar_[j] = 0
            roar_[j] = clf_shifted.predict(counterfactual_roar.reshape(1, -1))
            wachter_[j] = clf_shifted.predict(counterfactual_wachter.reshape(1, -1))

        drra_nm_m2.append(np.mean(drra_nm_))
        drra_gm_m2.append(np.mean(drra_gm_))
        ar_m2.append(np.mean(ar_))
        roar_m2.append(np.mean(roar_))
        wachter_m2.append(np.mean(wachter_))

    drra_nm_m1, drra_gm_m1, ar_m1, roar_m1, wachter_m1 = np.array(drra_nm_m1), np.array(drra_gm_m1), np.array(ar_m1), np.array(roar_m1), np.array(wachter_m1)
    drra_nm_m2, drra_gm_m2, ar_m2, roar_m2, wachter_m2 = np.array(drra_nm_m2), np.array(drra_gm_m2), np.array(ar_m2), np.array(roar_m2), np.array(wachter_m2)
    counterfactual_drra_nm_l, counterfactual_drra_gm_l, counterfactual_ar_l, counterfactual_roar_l, counterfactual_wachter_l = np.array(counterfactual_drra_nm_l), np.array(counterfactual_drra_gm_l), np.array(counterfactual_ar_l), np.array(counterfactual_roar_l), np.array(counterfactual_wachter_l)
    X_recourse_ = np.array(X_recourse_)

    validity['AR'] = [np.mean(ar_m1), np.std(ar_m1)] + [np.mean(ar_m2), np.std(ar_m2)] + cal_cost(counterfactual_ar_l, X_recourse_) + cal_cost(counterfactual_ar_l, X_recourse_, 'l2')
    validity['DiRRAc'] = [np.mean(drra_nm_m1), np.std(drra_nm_m1)] + [np.mean(drra_nm_m2), np.std(drra_nm_m2)] + cal_cost(counterfactual_drra_nm_l[:, :-1], X_recourse_) + cal_cost(counterfactual_drra_nm_l[:, :-1], X_recourse_, 'l2')
    validity['Gaussian DiRRAc'] = [np.mean(drra_gm_m1), np.std(drra_gm_m1)] + [np.mean(drra_gm_m2), np.std(drra_gm_m2)] + cal_cost(counterfactual_drra_gm_l[:, :-1], X_recourse_) + cal_cost(counterfactual_drra_gm_l[:, :-1], X_recourse_, 'l2')
    validity['ROAR'] = [np.mean(roar_m1), np.std(roar_m1)] + [np.mean(roar_m2), np.std(roar_m2)] + cal_cost(counterfactual_roar_l, X_recourse_) + cal_cost(counterfactual_roar_l, X_recourse_, 'l2')
    validity['Wachter'] = [np.mean(wachter_m1), np.std(wachter_m1)] + [np.mean(wachter_m2), np.std(wachter_m2)] + cal_cost(counterfactual_wachter_l, X_recourse_) + cal_cost(counterfactual_wachter_l, X_recourse_, 'l2')

    return validity
