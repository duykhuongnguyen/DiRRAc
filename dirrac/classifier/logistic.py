import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def logistic_classifier(X, y, intercept=False):
    """ Fit the data to a logistic regression model

    Args:
        X: inputs
        y: labels (binary)

    Returns:
        coef: parameters of model
    """
    clf = LogisticRegression(fit_intercept=intercept)
    clf.fit(X, y)

    # Retrieve the model parameters
    return clf, clf.coef_


def logistic_visualize(X, y, coef, b):
    """ Visualize logistic regression with the decision boundary

    Args:
        X: inputs
        y: labels (binary)
        coef: parameters of model
    """
    w1, w2 = coef.T
    # Calculate the intercept and gradient of the decision boundary
    c = -b / w2
    m = -w1 / w2

    # Plot the data and the classification with the decision boundary
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

    plt.scatter(*X[y == 0].T, s=8, alpha=0.5)
    plt.scatter(*X[y == 1].T, s=8, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')

    plt.show()
