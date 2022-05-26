from tqdm import tqdm

import numpy as np

import recourse as rs


class LinearAR(object):
    """ Class for generate counterfactual samples for framework: AR """

    def __init__(self, data, coef, intercept, padding=False, encoding_constraints=True, dis_indices=0, immutable_l=None, non_icr_l=None):
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
        
        if encoding_constraints:
            for i in range(dis_indices):
                self.action_set[str(i)].bounds = (float("-inf"), float("inf"))

        if padding:
            self.action_set[name_l[-1]].mutable = False

        if immutable_l:
            for i in immutable_l:
                self.action_set[str(i)].actionable = False

        if non_icr_l:
            for i in non_icr_l:
                self.action_set[str(i)].step_direction = 1

        self.action_set.set_alignment(coefficients=coef, intercept=intercept)
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
            fs = rs.Flipset(
                x=x,
                action_set=self.action_set,
                coefficients=self.coef,
                intercept=self.intercept,
            )
            fs_pop = fs.populate(total_items=1)

            counterfactual_sample = np.add(x, fs_pop.actions)
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
