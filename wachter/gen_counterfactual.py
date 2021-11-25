from tqdm import tqdm

import numpy as np

import torch

import gurobipy as grb


class Wachter(object):
    """ Class for generate counterfactual samples for framework: AR """

    def __init__(self, data, coef, intercept, lmbda=0.1, alpha=0.01, dist_type='l2', max_iter=20, padding=False):
        """ Parameters

        Args:
            data: data to generate counterfactuals
            model_trained: model trained on original data
            padding: True if we padding 1 at the end of instances
        """
        self.data = np.concatenate((data, np.ones(len(data)).reshape(-1, 1)), axis=1)
        self.coef = np.concatenate((coef, intercept))
        self.lmbda = lmbda
        self.alpha = alpha
        self.dim = self.data.shape[1]
        self.dist_type = dist_type
        self.max_iter = max_iter

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

    def fit_instance(self, x_0):
        x_t = torch.from_numpy(x_0.copy())
        x_t.requires_grad = True
        x_0 = torch.from_numpy(x_0)
        coef = torch.from_numpy(self.coef.copy())
        ord = None if self.dist_type=='l2' else 1
        g = 0

        for iter in range(self.max_iter):
            x_t.retain_grad()
            out = (1 / (1 + torch.exp(-torch.dot(coef, x_t))) - 1) ** 2 + self.lmbda * torch.linalg.norm(x_t - x_0, ord=ord)
            out.backward()
            g = x_t.grad
            x_t = x_t - self.alpha * g
            print(torch.dot(coef, x_t), 1 / (1 + torch.exp(-torch.dot(coef, x_t))))

            if torch.linalg.norm(self.alpha * g).item() < 1e-3:
                break

            if 1 / (1 + torch.exp(-torch.dot(coef, x_t))) >= 0.5:
                break

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
