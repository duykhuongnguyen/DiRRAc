from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn

import gurobipy as grb


class ROAR(object):
    """ Class for generate recourse for framework: ROAR """

    def __init__(self, data, coef, intercept, lmbda=0.1, delta_min=None, delta_max=0.1, alpha=0.1, dist_type=1, max_iter=20):
        """ Parameters

        Args:
            data: data to generate counterfactuals
            model_trained: model trained on original data
            padding: True if we padding 1 at the end of instances
        """
        self.data = data
        self.coef = coef
        self.intercept = intercept
        self.lmbda = lmbda
        self.alpha = alpha
        self.dim = self.data.shape[1]
        self.dist_type = dist_type
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.max_iter = max_iter

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
        model.params.IterationLimit = 1e3

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

    def fit_instance(self, x_0):
        x_0 = torch.tensor(x_0.copy()).float()
        x_t = x_0.clone().detach().requires_grad_(True)

        w = torch.from_numpy(self.coef.copy()).float()
        b = torch.tensor(self.intercept).float()
        y_target = torch.tensor([1]).float()
        lmbda = torch.tensor(self.lmbda).float()
        alpha = torch.tensor(self.alpha).float()
        loss_fn = nn.BCELoss()

        loss_diff = 1.0
        min_loss = float('inf')
        num_stable_iter = 0
        max_stable_iter = 5

        for it in range(self.max_iter):
            if x_t.grad is not None:
                x_t.grad.data.zero_()

            with torch.no_grad():
                lar_mul = self.delta_max / torch.sqrt(torch.linalg.norm(x_t) ** 2 + 1)
                delta_w = - x_t * lar_mul
                delta_b = - lar_mul
                w_ = w + delta_w
                b_ = b + delta_b

            f_x = torch.sigmoid(torch.dot(x_t, w_) + b_).float()
            cost = torch.dist(x_t, x_0, self.dist_type)
            f_loss = loss_fn(f_x, y_target)

            loss = f_loss + lmbda * cost
            loss.backward()

            with torch.no_grad():
                x_t -= alpha * x_t.grad

            loss_diff = min_loss - loss.data.item()
            if loss_diff <= 1e-4:
                num_stable_iter += 1
                if (num_stable_iter >= max_stable_iter):
                    break
            else:
                num_stable_iter = 0

            min_loss = min(min_loss, loss.data.item())
                
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

        for i in tqdm(range(l)):
            counterfactual_samples[i] = self.fit_instance(data[i])

        return counterfactual_samples
