from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Wachter(object):
    """ Class for generate recourse for framework: Wachter """

    def __init__(self, data, model, lmbda=0.1, lr=0.01, dist_type=1, max_iter=1000, decision_threshold=0.5, linear=False, encoding_constraints=False, cat_indices=0):
        """ Parameters

        Args:
            data: data to generate counterfactuals
            model_trained: model trained on original data
            padding: True if we padding 1 at the end of instances
        """
        self.data = data
        self.model = model

        if linear:
            self.coef = torch.tensor(self.model.coef_.squeeze()).float()
            self.intercept = torch.tensor(self.model.intercept_).float()
        self.lmbda = lmbda
        self.lr = lr

        self.dim = self.data.shape[1]
        self.dist_type = dist_type
        self.max_iter = max_iter
        self.decision_threshold = decision_threshold
        self.linear = linear
        self.encoding_constraints = encoding_constraints
        self.cat_indices = np.array(cat_indices).flatten()

    def fit_instance(self, x_0):
        x_0 = torch.from_numpy(x_0.copy()).float()
        x_t = Variable(x_0.clone(), requires_grad=True)
        x_enc = reconstruct_encoding_constraints(x_t, self.cat_indices)
        y_target = torch.tensor([1]).float()
        lmbda = torch.tensor(self.lmbda).float()
        f_x = self.model(x_enc) if not self.linear else torch.sigmoid(torch.dot(x_enc, self.coef) + self.intercept)

        loss_fn = torch.nn.BCELoss()
        optimizer = optim.Adam([x_t], self.lr, amsgrad=True)

        it = 0
        while f_x <= self.decision_threshold and it < self.max_iter:
            optimizer.zero_grad()

            if self.encoding_constraints:
                x_enc = reconstruct_encoding_constraints(x_t, self.cat_indices)
            else:
                x_enc = x_t.clone()

            f_x = self.model(x_enc) if not self.linear else torch.sigmoid(torch.dot(x_enc, self.coef) + self.intercept)

            cost = torch.dist(x_enc, x_0, self.dist_type)
            f_loss = loss_fn(f_x, y_target)

            loss = f_loss + lmbda * cost
            loss.backward()
            optimizer.step()
            it += 1

        return x_enc.cpu().detach().numpy().squeeze()

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


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc
