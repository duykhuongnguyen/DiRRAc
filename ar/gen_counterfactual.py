from tqdm import tqdm

import numpy as np

import recourse as rs


class LinearAR(object):
    """ Class for generate counterfactual samples for framework: AR """

    def __init__(self, data, coef, intercept, padding=False):
        """ Parameters

        Args:
            data: data to get upper bound, lower bound, action set
            coef: coefficients of classifier
            intercept: intercept of classifier
            padding: True if we padding 1 at the end of instances
        """
        self.n_variables = data.shape[1]

        # Action set
        name_l = [str(i) for i in range(self.n_variables)]
        self.action_set = rs.ActionSet(data, names = name_l)
        if padding:
            self.action_set[name_l[-1]].mutable = False
        self.coef = coef
        self.intercept = intercept

    def fit_instance(self, x):
        """ Fit linear recourse action with an instance

        Args:
            x: a single input

        Returns:
            counterfactual_sample: counterfactual of input x
        """
        try:
            rb = rs.RecourseBuilder(
                coefficients=self.coef,
                intercept=self.intercept,
                action_set=self.action_set,
                x=x
            )
            output = rb.fit()
            counterfactual_sample = np.add(x, output['actions'])
        except:
            counterfactual_sample = np.zeros(x.shape)

        return counterfactual_sample

    def fit_data(self, data):
        """ Fit linear recourse action with all instances

        Args:
            data: all the input instances

        Returns:
            counterfactual_samples: counterfactual of instances in dataset
        """
        l = len(data)
        counterfactual_samples = np.zeros((l, self.n_variables))

        for i in tqdm(range(l)):
            counterfactual_samples[i] = self.fit_instance(data[i])

        return counterfactual_samples
