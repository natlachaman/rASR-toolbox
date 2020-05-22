import pymanopt
import numpy as np
from scipy import linalg

from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.solvers import TrustRegions

from .helpers.utils import _mldivide


def nonlinear_eigenspace(L: np.ndarray, k: float, alpha: int = 1) -> (np.ndarray, np.ndarray):
    """Nonlinear eigenvalue problem: total energy minimization.
    This example is motivated in [1]_ and was adapted from the manopt toolbox
    in Matlab.

    Parameters
    ----------
    L : np.ndarray
        Discrete Laplacian operator: the covariance matrix of shape=(n_channels, n_channels)
    alpha : float
        Given constant for optimization problem.
    k : int (default: 1)
        Determines how many eigenvalues are returned.

    Returns
    -------
    Xsol : np.ndarray of shape=(n_channels, n_channels)
        Eigenvectors.
    S0 : np.ndarray
        Eigenvalues.

    References
    ----------
    .. [1] "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
       Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin, SIAM Journal on Matrix
       Analysis and Applications, 36(2), 752-774, 2015.

    """
    n = L.shape[0]
    assert L.shape[1] == n, 'L must be square.'

    # Grassmann manifold description
    manifold = Grassmann(n, k)
    # manifold._dimension = 1  # hack

    # A solver that involves the hessian (check if correct TODO)
    solver = TrustRegions()

    # Cost function evaluation
    def cost(X):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # diag(X*X')
        val = 0.5 * np.trace(X.T @ (L * X)) + (alpha / 4) * (rhoX.T @ _mldivide(L, rhoX))
        return val

    # Euclidean gradient evaluation
    # @pymanopt.function.Callable
    def egrad(X):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # diag(X*X')
        g = L @ X + alpha * np.diagflat(_mldivide(L, rhoX)) @ X
        return g

    # Euclidean Hessian evaluation
    # Note: Manopt automatically converts it to the Riemannian counterpart.
    # @pymanopt.function.Callable
    def ehess(X, U):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # np.diag(X * X')
        rhoXdot = 2 * np.sum(X.dot(U), 1)
        h = L @ U + alpha * np.diagflat(_mldivide(L, rhoXdot)) @ X + alpha * np.diagflat(_mldivide(L, rhoX)) @ U
        return h

    # Initialization as suggested in above referenced paper.
    # randomly generate starting point for svd
    x = np.random.randn(n, k)
    [U, S, V] = linalg.svd(x, full_matrices=False)
    x = U.dot(V.T)
    S0, U0 = linalg.eig(L + alpha * np.diagflat(_mldivide(L, np.sum(x**2, 1))))

    # Call manoptsolve to automatically call an appropriate solver.
    # Note: it calls the trust regions solver as we have all the required
    # ingredients, namely, gradient and Hessian, information.
    # todo: UnboundLocalError: local variable 'j' referenced before assignment
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=0)
    Xsol = solver.solve(problem, x=U0, maxinner=4)

    return Xsol, S0