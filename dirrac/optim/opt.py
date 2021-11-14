import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad

from autograd.scipy.stats import norm

import gurobipy as grb


class Optimization(object):
    """ Class for optimization problem """

    def __init__(self, delta_add, K, dim, p, theta, sigma, rho, lmbda, zeta, dist_type='l2', gaussian=False, model_type='mixture', real_data=False, num_discrete=None, padding=False):
        self.delta_add = delta_add
        self.K = K
        self.dim = dim
        self.p = p
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lmbda = lmbda
        self.zeta = zeta
        self.dist_type = dist_type
        self.gaussian = gaussian
        self.model_type = model_type
        self.real_data = real_data
        self.num_discrete = num_discrete
        self.padding = padding

        self.df_autograd = value_and_grad(self.f_moments_infor) if not gaussian else value_and_grad(self.f_gaussian)

    def find_delta_min(self, x_0):
        """ Find delta min with each instance

        Args:
            x_0: original input
        """
        # Model initialization
        model = grb.Model("qcp")
        model.params.NonConvex = 2
        model.setParam('OutputFlag', False)
        model.params.threads = 64

        # Variables
        x = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="x")
        x_sub_0 = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="xsub0")
        x_norm = model.addMVar(1, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="x_norm")
        x_sub_0_abs = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="xsub0abs")

        # Set objective
        if self.dist_type == 'l2':
            obj = x_sub_0 @ x_sub_0
        elif self.dist_type == 'l1':
            obj = x_sub_0_abs.sum()
        else:
            raise ValueError('Invalid distance type')

        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Constraints
        model.addConstr(x_sub_0 == x - x_0)
        model.addConstr(x_norm @ x_norm == x @ x)
        model.addConstr(x_norm >= 0)

        # If x is real data it need to be greater than 0 and less than or equal to 1
        if self.real_data:
            model.addConstr(x >= 0)
            model.addConstr(x <= 1)

        if self.padding:
             model.addConstr(x[self.dim - 1] == 1)

        if self.dist_type == 'l1':
            # for w, v in zip(x_sub_0_abs.tolist(), x_sub_0.tolist()):
                # model.addConstr(w == grb.abs_(v))
            model.addConstr(x_sub_0 @ x_sub_0 == x_sub_0_abs @ x_sub_0_abs)
            model.addConstr(x_sub_0_abs >= 0)

        for k in range(self.K):
            model.addConstr(-self.theta[k].T @ x + self.rho[k] * x_norm <= 0)   # Constrant

        # Optimize
        model.optimize()

        x_opt = np.zeros(self.dim)

        for i in range(self.dim):
            x_opt[i] = x[i].x
        delta_min = np.linalg.norm(x_opt - x_0) if self.dist_type == 'l2' else np.linalg.norm(x_opt - x_0, ord=1)

        return delta_min

    def f_moments_infor(self, x, model_type='mixture'):
        """ Function f

        Args:
            x: input

        Returns:
            f_val: value of function f
        """
        f_val = 0
        for k in range(self.K):
            A_k = np.dot(-self.theta[k].T, x)
            B_k = np.sqrt(np.dot(np.dot(x.T, self.sigma[k]), x))
            C_k = self.rho[k] * np.linalg.norm(x)
            numerator = -A_k * C_k + B_k * np.sqrt(A_k ** 2 + B_k ** 2 - C_k ** 2)
            denominator = A_k ** 2 + B_k ** 2

            if self.model_type == 'mixture':
                f_val += self.p[k] * (numerator / denominator) ** 2
            elif self.model_type == 'worst_case':
                f_val = max(numerator / denominator, f_val)
            else:
                raise ValueError('Invalid model type')

        return f_val

    def f_gaussian(self, x, model_type='mixture'):
        """ Function f if gaussian distribution

        Args:
            input

        Returns:
            1 - f_val: value of funtion f-gaussian
        """
        f_val = 0
        for k in range(self.K):
            A_k = np.dot(-self.theta[k].T, x)
            B_k = np.sqrt(np.dot(np.dot(x.T, self.sigma[k]), x))
            C_k = self.rho[k] * np.linalg.norm(x)
            numerator = A_k ** 2 - C_k ** 2
            denominator = -A_k * B_k + C_k * np.sqrt(A_k ** 2 + B_k ** 2 - C_k ** 2)

            if self.model_type == 'mixture':
                f_val += self.p[k] * norm.cdf((numerator / denominator))
            elif self.model_type == 'worst_case':
                f_val = max(1 - norm.cdf((numerator / denominator)), f_val)
            else:
                raise ValueError('Invalid model type')

        return 1 - f_val if self.model_type == 'mixture' else f_val

    def projection_moments_infor(self, x_0, x_comma, delta, check_feasible=False):
        """ Projected gradient

        Args:
            K: number of mixtures
            theta, rho: mixture params
            x_comma: input after gradient
            x_0: original input
            delta: paramter

        Returns:
            x_opt: optimal value of x
        """
        # Model initialization
        model = grb.Model("qcp")
        model.params.NonConvex = 2
        model.setParam('OutputFlag', False)
        model.params.threads = 64

        # Variables
        x = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="x")
        x_sub_comma = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="xsubcomma")
        x_sub_0 = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="xsub0")
        x_norm = model.addMVar(1, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="x_norm")
        x_sub_0_abs = model.addMVar(self.dim, lb=float('-inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, name="xsub0abs")

        # Set objective
        obj = x_sub_comma @ x_sub_comma if not check_feasible else 0
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Constraints
        model.addConstr(x_sub_comma == x - x_comma)
        model.addConstr(x_sub_0 == x - x_0)
        model.addConstr(x_norm @ x_norm == x @ x)
        model.addConstr(x_norm >= 0)

        # If x is real data it need to be greater than 0 and less than or equal to 1
        if self.real_data:
            model.addConstr(x >= 0)
            model.addConstr(x <= 1)

        if self.padding:
            model.addConstr(x[self.dim - 1] == 1)

        if self.dist_type == 'l1':
            # for w, v in zip(x_sub_0_abs.tolist(), x_sub_0.tolist()):
            #     model.addConstr(w == grb.abs_(v))
            model.addConstr(x_sub_0 @ x_sub_0 == x_sub_0_abs @ x_sub_0_abs)
            model.addConstr(x_sub_0_abs >= 0)
            model.addConstr(sum(x_sub_0_abs) <= delta)
        else:
            model.addConstr(x_sub_0 @ x_sub_0 <= delta * delta)     # Constrant 1

        for k in range(self.K):
            model.addConstr(-self.theta[k].T @ x + self.rho[k] * x_norm <= -1e-3)   # Constrant 

        # Optimize
        model.optimize()

        x_opt = np.zeros(self.dim)

        for i in range(self.dim):
            x_opt[i] = x[i].x

        return x_opt

    def find_optimal_k(self, x_t, x_0, delta):
        """ Find value of k at each step

        Args:
            x_t: input at step t
            x_0: input
            lmbda, zeta, delta: hyperparams

        Returns:
            x_opt: x at step t + 1
        """
        i = 0
        f_x, df_x = self.df_autograd(x_t)
        if f_x < 1e-3:
            return x_t, f_x, f_x

        while True:
            x_comma = x_t - (self.lmbda ** i) * self.zeta * df_x
            x_opt = self.projection_moments_infor(x_0, x_comma, delta)
            f_x_opt, df_x_opt = self.df_autograd(x_opt)
            if f_x_opt <= f_x - 1 / (2 * self.lmbda ** i * self.zeta) * np.dot(x_t - x_opt, x_t - x_opt):
                return x_opt, f_x_opt, f_x

            if i >= 1000:
                return x_opt, f_x_opt, f_x
            i += 1

    def recourse_action(self, x_0, max_iter):
        """ Full process of recource action """
        try:
            delta_min = self.find_delta_min(x_0)
        except:
            return

        delta = delta_min + self.delta_add
        # Check if feasible set is non-empty
        try:
            self.projection_moments_infor(x_0, x_0, delta, check_feasible=True)
        except:
            return

        # Initialization step for x_hat_t
        x_t = self.projection_moments_infor(x_0, x_0, delta)

        # Iter till converge
        for iter in range(max_iter):
            x_opt, f_x_opt, f_x = self.find_optimal_k(x_t, x_0, delta)
            x_t = x_opt
            if abs(f_x_opt - f_x) < 1e-3:
                break

        return f_x_opt, x_opt
