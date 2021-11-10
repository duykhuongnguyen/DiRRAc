import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad

import torch

import gurobipy as grb


class ROAR(object):
    """ Class for generate counterfactual samples for framework: AR """

    def __init__(self, data, model_trained, lmbda=0.1, sigma_min=None, sigma_max=0.5, alpha=0.1, dist_type='l2', padding=False):
        """ Parameters

        Args:
            data: data to generate counterfactuals
            model_trained: model trained on original data
            padding: True if we padding 1 at the end of instances
        """
        self.data = np.concatenate((data, np.ones(len(data)).reshape(-1, 1)), axis=1)
        self.coef = np.concatenate((model_trained.coef_.squeeze(), model_trained.intercept_))
        self.lmbda = lmbda
        self.alpha = alpha
        self.dim = self.data.shape[1]
        self.dist_type = dist_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def objective_func(self, coef, x, x_0):
        """ Loss function - mse or log loss

        Args:
            coef: model params
            x: a single input
            x_0; original input
            loss_type: mse or log loss
            dist_type: l1 or l2

        Returns:
            output: output of objective function
        """
        dist = torch.linalg.norm(x - x_0)
        loss = (torch.dot(coef, x) - 1) ** 2
        output = loss + self.lmbda * dist
        return output

    def find_optimal_sigma(self, coef, x):
        """ Find value of sigma at each step

        Args:
            coef: coef of model
            x: input

        Returns:
            x_opt: x at step t + 1
        """
        # Model initialization
        model = grb.Model("qcp")
        model.params.NonConvex = 2
        model.setParam('OutputFlag', False)
        model.params.threads = 64

        sigma = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="sigma")
        sigma_norm = model.addMVar(1, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="sigma_norm")

        # Set objective
        obj = x @ sigma + np.dot(x, coef)
        model.setObjective(obj, grb.GRB.MAXIMIZE)

        # Constraints
        if self.sigma_min:
            model.addConstr(self.sigma_min <= sigma)
            model.addConstr(sigma <= self.sigma_max)
        else:
            model.addConstr(sigma_norm @ sigma_norm == sigma @ sigma)
            model.addConstr(sigma_norm <= self.sigma_max)
            model.addConstr(sigma_norm >= 0)

        model.optimize()

        sigma_hat = np.zeros(self.dim)

        for i in range(self.dim):
            sigma_hat[i] = sigma[i].x

        return sigma_hat


    def fit_instance(self, x_0, max_iter):
        x_t = torch.from_numpy(x_0.copy())
        x_t.requires_grad = True
        x_0 = torch.from_numpy(x_0)
        coef = torch.from_numpy(self.coef.copy())
        coef_ = torch.from_numpy(self.coef.copy())
        ord = None if self.dist_type=='l2' else 1
        g = 0

        for iter in range(max_iter):
            sigma_hat = self.find_optimal_sigma(coef, x_t.detach().numpy())
            coef_ = coef + torch.from_numpy(sigma_hat)
            x_t.retain_grad()
            out = (torch.dot(coef_, x_t) - 1) ** 2 + self.lmbda * torch.linalg.norm(x_t - x_0, ord=ord)
            out.backward()
            g = x_t.grad
            x_t = x_t - self.alpha * g

        return x_t.detach().numpy()


    def fit_data(self, data):
        """ Fit linear recourse action with all instances

        Args:
            data: all the input instances

        Returns:
            counterfactual_samples: counterfactual of instances in dataset
        """
        l = len(data)
        counterfactual_samples = np.zeros((l, self.dim))

        for i in range(l):
            counterfactual_samples[i] = self.fit_instance(data[i], 10)

        return counterfactual_samples
