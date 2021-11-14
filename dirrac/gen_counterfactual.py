from tqdm import tqdm

import numpy as np

from dirrac.optim.opt import Optimization


class DRRA(object):

    def __init__(self, delta_add, k, dim, p, theta, sigma, rho, lmbda, zeta, dist_type='l2', real_data=False, num_discrete=None, padding=False):
        self.delta_add = delta_add
        self.k = k
        self.dim = dim
        self.p = p
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lmbda = lmbda
        self.zeta = zeta
        self.dist_type = dist_type
        self.real_data = real_data
        self.padding = padding
        self.num_discrete = num_discrete

        self.nm = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding)
        self.nwc = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, model_type='worst_case', real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding)
        self.gm = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, gaussian=True, real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding)
        self.gwc = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, gaussian=True, model_type='worst_case', real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding)
        self.models = {'nm': self.nm, 'nwc': self.nwc, 'gm': self.gm, 'gwc': self.gwc}

    def fit_instance(self, x, model='nm'):
        """ Recorse action with an instance

        Args:
            x: original instance
            model: model type

        Returns:
            x_opt: recourse of x
        """
        out = self.models[model].recourse_action(x, 10)
        try:
            f_opt, x_opt = out
        except:
            x_opt = x.copy()

        return x_opt

    def fit_data(self, data, model='nm'):
        counterfactual_samples = np.zeros((len(data), self.dim))
        for i in tqdm(range(len(data))):
            counterfactual_samples[i] = self.fit_instance(data[i], model=model)

        return counterfactual_samples
