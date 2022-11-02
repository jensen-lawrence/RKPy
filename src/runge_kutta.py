# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from math import isclose
from typing import Any, Union
# from sympy import Rational

# ------------------------------------------------------------------------------
# RungeKutta Class
# ------------------------------------------------------------------------------

class RungeKutta:
    """
    Class for explicit and implicit Runge-Kutta methods with fixed or adaptive
    step sizes. A Runge-Kutta method is uniquely characterized by the
    parameters A, b, c, where:
    - A is the nxn-dimensional Runge-Kutta matrix.
    - For fixed step methods, b is the n-dimensional vector of Runge-Kutta
    weights; for adaptive methods, b is the tuple (b1, b2), where b1 is the
    n-dimensional vector of weights for the primary method, and b2 is the
    n-dimensional vector of weights for the embedded method used in error
    calculation and step size correction.
    - c is the n-dimensional vector of Runge-Kutta nodes. If no c is provided,
    it is calculated automatically as $c_i = \sum_j A_{ij}$. If c is specified
    by the user, the summation condition is verified.
    """

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------

    def __init__(self, name: str, A: np.ndarray,
                 b: Union[np.ndarray, tuple[np.ndarray]],
                 c: np.ndarray = None) -> None:
        """
        Initializes the RungeKutta class.

        Parameters
        ----------
        name : str
            The name of the fixed Runge-Kutta method.
        A : numpy.ndarray, shape=(n,n)
            The Runge-Kutta matrix.
        b : Union[numpy.ndarray, tuple[numpy.ndarray]], shape=(n,)
            For fixed step size methods, the vector of Runge-Kutta weights.
            For adaptive step size methods, the tuple (b1, b2), where b1 is
            the vector of weights for the primary method and b2 is the vector
            of weights for the embedded method used in error calculation
            and step size correction.
        c : numpy.ndarray, shape=(n,), optional
            The vector of Runge-Kutta nodes. Set to None by default and
            automatically calculated as $c_i = \sum_j A_{ij}$.

        Returns
        -------
        None
        """
        self._name = name
        self._A = A
        self._b = b
        self._c = c
        self._check_parameters()


    def __str__(self) -> None:
        """
        Provides a printable representation of the RungeKutta class.

        Returns
        -------
        None
        """
        rep_str = f"{self.name}\n"
        expl_str = "Explicit" if self.isexplicit else "Implicit"
        adpt_str = "adaptive" if self.isadaptive else "fixed"
        rep_str += f"{expl_str}, {adpt_str} Runge-Kutta method"
        rep_str += f" with {self.n_stages} stages\n"
        rep_str += self.tableau()
        return rep_str


    def __eq__(self, other: Any) -> bool:
        """
        Determines whether an instance of the RungeKutta class is equal
        to another object.

        Parameters
        ----------
        other : Any
            Object being compared to an instance of the RungeKutta class.

        Returns
        -------
        bool
            True if other is equal to self, False otherwise.
        """
        class_match = isinstance(other, RungeKutta)
        param_match = (self.A == other.A) and (self.b == other.b) \
                      and (self.c == other.c)
        return class_match and param_match


    def _check_parameters(self) -> None:
        """
        Determines whether the parameters provided to the RungeKutta
        class are valid parameters.

        Returns
        -------
        None

        Raises
        ------
        TypeError
        - name is not a str
        - A and c are not np.ndarrays
        - b is not an np.ndarray for fixed step methods
        - b is not a tuple of np.ndarrays for adaptive methods

        ValueError
        - A is not a square matrix
        - c is not a vector (one-dimensional)
        - The rows of A do not sum to the elements of c
        - b is not a vector (fixed step methods)
        - b1 and b2 are not vectors (adaptive methods)
        - The elements of b do not sum to 1 (fixed step methods)
        - The elements of b1 and b2 do not sum to 1 (adaptive methods)
        - The dimensions of A, b, and c are inconsistent (fixed step methods)
        - The dimensions of A, b1, b2, and c are inconsistent (adaptive methods)
        """
        # Verify name is a string
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, not {type(self.name)}")

        # Verify A and c and np.ndarrays
        for arr in zip(("A", "c"), (self.A, self.c)):
            if not isinstance(arr[1], np.ndarray):
                raise TypeError(f"{arr[0]} must be a numpy.ndarray, "
                                f"not {type(arr[1])}")

        # Verify A is a square matrix
        if (self.A.shape[0] != self.A.shape[1]) or (len(self.A.shape) > 2):
            raise ValueError("Invalid dimensions: A must have dimensions "
                             "(n, n), but instead has dimensions "
                             f"{self.A.shape}")

        # Checks for if c is provided by the user
        if self._c is not None:
            # Verify c is a vector
            if len(self.c.shape) > 1:
                raise ValueError("Invalid dimensions: c must have dimensions "
                                "(n,), but instead has dimensions "
                                f"{self.c.shape}")

            # Verify the rows of A sum to the elements of c
            bad_idxs = np.array([i for i in range(self.c.size)
                                if not isclose(np.sum(self.A[i,:]), self.c[i],
                                                      abs_tol=1e-8)])
            bad_Ais = np.array([np.sum(self.A[i,:]) for i in bad_idxs])
            bad_cs = np.array([self.c[i] for i in bad_idxs])
            if bad_idxs.size > 0:
                raise ValueError("Invalid matrix or nodes: the rows of A must "
                                 "sum to the elements of c, but row(s) "
                                 f"{bad_idxs} of A sum to {bad_Ais}, while "
                                 f"element(s) {bad_idxs} of c are {bad_cs}")

        # Additional checks for adaptive methods
        if self.isadaptive:
            for arr in zip(("b1", "b2"), (self.b1, self.b2)):
                # Verify b1 and b2 are np.ndarrays
                if not isinstance(arr[1], np.ndarray):
                    raise TypeError(f"{arr[0]} must be a numpy.ndarray, "
                                    f"not {type(arr[1])}")

                # Verify b1 and b2 are vectors
                if len(arr[1].shape) > 1:
                    raise ValueError(f"Invalid dimensions: {arr[0]} must have "
                                     "dimensions (n,), but instead has "
                                     f"dimensions {arr[1].shape}")

                # Verify the elements of b1 and b2 sum to 1
                if not isclose(np.sum(arr[1]), 1.0, rel_tol=1e-8):
                    raise ValueError("Invalid weights: the elements of "
                                     f"{arr[0]} must sum to 1, but instead "
                                     f"sum to {np.sum(arr[1])}")

            # Verify the dimensions of A, b1, b2, and c match
            if not self.A.shape[0] == self.b1.size == \
                   self.b2.size == self.c.size:
                raise ValueError("Invalid dimensions: dim(A), dim(b1), "
                                 "dim(b2), and dim(c) must be (n, n), n, "
                                 f"and n, but instead dim(A) = {self.A.shape}, "
                                 f"dim(b1) = {self.b1.size}, dim(b2) = "
                                 f"{self.b2.size}, and dim(c) = {self.c.size}")

        # Additional checks for fixed methods
        else:
            # Verify b is an np.ndarray
            if not isinstance(self.b, np.ndarray):
                raise TypeError("b must be a numpy.ndarray, not "
                                f"{type(self.b)}")

            # Verify b is a vector
            if len(self.b.shape) > 1:
                raise ValueError("Invalid dimensions: b must have "
                                 "dimensions (n,), but instead has dimensions "
                                 f"{self.b.shape}")

            # Verify the elements of b sum to 1
            if not isclose(np.sum(self.b), 1.0, rel_tol=1e-8):
                raise ValueError(f"Invalid weights: the elements of b must sum "
                                 f"to 1, but instead sum to {np.sum(self.b)}")

            # Verify the dimensions of A, b, and c match
            if not self.A.shape[0] == self.b.size == self.c.size:
                raise ValueError("Invalid dimensions: dim(A), dim(b), and "
                                 "dim(c) must be (n, n), n, and n, but instead "
                                 f"dim(A) = {self.A.shape}, dim(b) = "
                                 f"{self.b.size}, and dim(c) = {self.c.size}")

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    def tableau(self, output: str = None) -> str:
        """
        Determines the Butcher tableau of the Runge-Kutta method.

        Parameters
        ----------
        output : str, optional
            Path and file name to where the Butcher tableau will be saved.
            None by default. If left unspecified, the tableau is not saved.

        Returns
        -------
        str
            String representing the Butcher tableau.
        """
        # Control spacing in tableau
        spaces = np.zeros(self.n_stages + 1, dtype=int)
        spaces[0] = max([len(str(ci)) for ci in self.c])

        if self.isadaptive:
            for i in range(self.n_stages):
                spaces[i+1] = max([len(str(aij)) for aij in self.A[:,i]] \
                            + [len(str(self.b1[i]))] + [len(str(self.b2[i]))])

        else:
            for i in range(self.n_stages):
                spaces[i+1] = max([len(str(aij)) for aij in self.A[:,i]] \
                            + [len(str(self.b[i]))])
            
        # Initialize tableau string and add c_i rows
        tableau = ""
        for i in range(self.n_stages):
            row = ""
            row += f"{self.c[i]}" + " "*(spaces[0] - len(str(self.c[i]))) + " |"
            for j in range(self.n_stages):
                row += f" {self.A[i,j]}" + " "*(spaces[j+1] \
                    - len(str(self.A[i,j])))
            row += "\n"
            tableau += row
            
            if i == self.n_stages - 1:
                tableau += "-"*(len(row) - 1)
                tableau += "\n"

        # Add b rows for adaptive methods
        if self.isadaptive:
            tableau += " "*spaces[0] + " |"
            for i in range(self.n_stages):
                tableau += f" {self.b1[i]}" + " "*(spaces[i+1] \
                        - len(str(self.b1[i])))
            tableau += "\n"
            tableau += " "*spaces[0] + " |"
            for i in range(self.n_stages):
                tableau += f" {self.b2[i]}" + " "*(spaces[i+1] \
                        - len(str(self.b2[i])))

        # Add b row for fixed methods
        else:
            tableau += " "*spaces[0] + " |"
            for i in range(self.n_stages):
                tableau += f" {self.b[i]}" + " "*(spaces[i+1] \
                        - len(str(self.b[i])))

        # Save result
        if output is not None:
            with open(output, "w") as f:
                f.write(f"{self.name} Butcher tableau:\n")
                f.write(tableau)
            f.close()

        return tableau

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Public property for _name."""
        return self._name


    @property
    def A(self) -> np.ndarray:
        """Public property for _A."""
        return self._A


    @property
    def b(self) -> Union[np.ndarray, tuple[np.ndarray]]:
        """Public property for _b."""
        return self._b


    @property
    def b1(self) -> np.ndarray:
        """
        Property for b1, the vector of Runge-Kutta weights for the primary
        method in an adaptive Runge-Kutta method. Raises an AttributeError
        if called on a fixed step method.

        Returns
        -------
        np.ndarray
            The vector of primary Runge-Kutta weights b1.

        Raises
        ------
        AttributeError
            When called on a fixed step step Runge-Kutta method.
        """
        if self.isadaptive:
            return self.b[0]
        else:
            raise AttributeError(f"{self.name} is not an adaptive Runge-Kutta "
                                 "method and has no attribute 'b1'")


    @property
    def b2(self) -> np.ndarray:
        """
        Property for b2, the vector of Runge-Kutta weights for the embedded
        method in an adaptive Runge-Kutta method. Raises an AttributeError
        if called on a fixed step method.

        Returns
        -------
        np.ndarray
            The vector of embedded Runge-Kutta weights b2.

        Raises
        ------
        AttributeError
            When called on a fixed step step Runge-Kutta method.
        """
        if self.isadaptive:
            return self.b[1]
        else:
            raise AttributeError(f"{self.name} is not an adaptive Runge-Kutta "
                                 "method and has no attribute 'b2'")


    @property
    def c(self) -> np.ndarray:
        """Public property for _c. Calculated automatically if c is None."""
        if self._c is not None:
            return self._c
        else:
            return np.array([np.sum(self.A[i,:])
                             for i in range(self.A.shape[0])])


    @property
    def n_stages(self) -> int:
        """The number of stages the Runge-Kutta method has."""
        return self.c.size


    @property
    def isexplicit(self) -> bool:
        """Whether the Runge-Kutta method is explicit."""
        return np.sum(self.A[0,:]) == 0.0


    @property
    def isimplicit(self) -> bool:
        """Whether the Runge-Kutta method is implicit."""
        return not self.isexplicit


    @property
    def isadaptive(self) -> bool:
        """Whether the Runge-Kutta method is adaptive."""
        if isinstance(self.b[0], np.ndarray):
            return True
        else:
            return False

# ------------------------------------------------------------------------------
# RK Function
# ------------------------------------------------------------------------------

def RK(name: str, A: np.ndarray, b: Union[np.ndarray, tuple[np.ndarray]],
       c: np.ndarray = None) -> RungeKutta:
    """
    Wrapper function for the RungeKutta class to provide RK as an abbreviation
    of RungeKutta.
    """
    return RungeKutta(name, A, b, c=c)
    