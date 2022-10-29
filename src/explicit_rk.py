# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from math import isclose
from time import time
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Explicit RK class
# ------------------------------------------------------------------------------

class ExplicitRK:
    """
    Implementation of a generic, n-stage, explicit Runge-Kutta method.
    """
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """
        Initializes the ExplicitRK class.

        Parameters
        ----------
        A : numpy.ndarray, shape=(n,n)
            The Runge-Kutta matrix
        b : numpy.ndarray, shape=(n,)
            The vector of Runge-Kutta weights
        c : numpy.ndarray, shape=(n,)
            The vector of Runge-Kutta nodes

        Returns
        -------
        None
        """
        # Verify inputs are np.ndarrays
        for arr in zip(("A", "b", "c"), (A, b, c)):
            if not isinstance(arr[1], np.ndarray):
                raise TypeError(f"{arr[0]} must be a numpy.ndarray, "
                                f"not {type(arr[1])}")

        # Verify inputs have the correct dimensions
        if (A.shape[0] != A.shape[1]) or (len(A.shape) > 2):
            raise ValueError("Invalid dimensions: A must have dimensions "
                             f"(n, n), but instead has dimensions {A.shape}")

        for arr in zip(("b", "c"), (b, c)):
            if len(arr[1].shape) > 1:
                raise ValueError(f"Invalid dimensions: {arr[0]} must have "
                                 "dimensions (n,), but instead has dimensions "
                                 f"{arr[1].shape}")

        if not A.shape[0] == b.size == c.size:
            raise ValueError("Invalid dimensions: dim(A), dim(b), and dim(c) "
                             "must be (n,n), n, and n, but instead dim(A) = "
                             f"{A.shape}, dim(b) = {b.size}, and dim(c) = "
                             f"{c.size}")

        # Verify inputs meet requirements for an explicit RK method
        if not isclose(np.sum(b), 1.0):
            raise ValueError("Invalid weights: the elements of b must sum "
                             f"to 1, but instead sum to {np.sum(b)}")

        if c[0] != 0.0:
            raise ValueError("Invalid nodes: c[0] must be 0 for the "
                             "Runge-Kutta method to be explicit, but instead "
                             f"is {c[0]}")

        bad_idxs = np.array([i for i in range(c.size)
                            if not isclose(np.sum(A[i,:]), c[i])])
        bad_Ais = np.array([np.sum(A[i,:]) for i in bad_idxs])
        bad_cs = np.array([c[i] for i in bad_idxs])
        if bad_idxs.size > 0:
            raise ValueError("Invalid matrix or nodes: the rows of A must sum "
                             f"to the elements of c, but row(s) {bad_idxs} "
                             f"of A sum to {bad_Ais}, while element(s) "
                             f"{bad_idxs} of c are {bad_cs}")
        
        # Initialize class attributes
        self.A = A
        self.b = b
        self.c = c
        self.n = b.size
