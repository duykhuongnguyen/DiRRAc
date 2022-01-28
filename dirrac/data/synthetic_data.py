import numpy as np


rng = np.random.default_rng(seed=12)


class DataSynthesizer(object):
    """ Class for data synthesize """

    def __init__(self, mean_0, cov_0, mean_1, cov_1, n):
        """ Parameters

        Args:
            mean_0: mean of class 0 of initial data distribution
            cov_0: mean of class 0 of initial data distribution
            mean_1: mean of class 0 of initial data distribution
            cov_1: mean of class 0 of initial data distribution
            n: num samples

        """
        self.mean_0 = mean_0
        self.cov_0 = cov_0
        self.mean_1 = mean_1
        self.cov_1 = cov_1
        self.n = n

    def synthesize_data(self, mean_0, cov_0, mean_1, cov_1):
        """ Synthesize data with binary class with Gaussian distribution

        Args:
            mean_0: mean of distribution 0
            cov_0: covariance of distribution 0
            mean_1: mean of distribution 1
            cov_1: covariance of distribution 1
            n: num samples

        Returns:
            X: features
            y: labels
        """
        x_0 = rng.multivariate_normal(mean_0, cov_0, self.n // 2)
        x_1 = rng.multivariate_normal(mean_1, cov_1, self.n // 2)
        X = np.concatenate((x_0, x_1))

        y = np.concatenate((np.zeros(self.n // 2), np.ones(self.n // 2)))

        return X, y

    def synthesize_modes_data(self, num_iter, p, factor, same='True'):
        """ Synthesize initial data and shift data

        Args:
            num_iter: number of dataset to be generated
            p: prob of each mode to be generated
            factor: factor of each mode
        """
        # Generate initial data
        X_initial, y_initial = self.synthesize_data(self.mean_0, self.cov_0, self.mean_1, self.cov_1)

        # Generate each mode data
        samples_mode = [int(num_iter * p[i]) for i in range(len(p))]
        features = np.zeros((1 + sum(samples_mode), self.n, X_initial.shape[1]))
        labels = np.zeros((1 + sum(samples_mode), self.n))
        features[0] = X_initial
        labels[0] = y_initial

        # Generate mean shift data
        i = 1
        for j in range(samples_mode[0]):
            const = [factor[0] * (j + 1), 0] if not same else [1.5, 0]
            X, y = self.synthesize_data(self.mean_0 + const, self.cov_0, self.mean_1, self.cov_1)
            features[i] = X
            labels[i] = y
            i += 1

        # Generate cov shift data
        for j in range(samples_mode[1]):
            const = (3 + factor[1] * (j + 1)) * self.cov_0 if not same else 4 * self.cov_0
            X, y = self.synthesize_data(self.mean_0, const, self.mean_1, self.cov_1)
            features[i] = X
            labels[i] = y
            i += 1

        # Generate cov shift data
        for j in range(samples_mode[2]):
            const1 = [factor[0] * (j + 1), 0] if not same else [1.5, 0]
            const2 = (3 + factor[1] * (j + 1)) * self.cov_0 if not same else 4 * self.cov_0
            X, y = self.synthesize_data(self.mean_0 + const1, const2, self.mean_1, self.cov_1)
            features[i] = X
            labels[i] = y
            i += 1

        return features, labels


if __name__ == '__main__':
    mean_0 = np.ones(2) * (-2)
    mean_1 = np.ones(2) * 2
    cov_0 = cov_1 = 0.5 * np.identity(2)
    n = 1000

    sd = DataSynthesizer(mean_0, cov_0, mean_1, cov_1, n)
    X, y = sd.synthesize_modes_data(100, [0.2, 0.4, 0.4], [0.1, 0.1])
    print(X.shape, y.shape, y[0])
