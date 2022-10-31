# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from math import isclose
from time import time
from tqdm import tqdm
from typing import Any, Callable, Union

# ------------------------------------------------------------------------------
# Explicit RK class
# ------------------------------------------------------------------------------

class ExplicitRK:
    """
    Implementation of a generic, n-stage, explicit Runge-Kutta method.
    """
    def __init__(self, name: str, A: np.ndarray, b: np.ndarray,
                 c: np.ndarray) -> None:
        """
        Initializes the ExplicitRK class.

        Parameters
        ----------
        name : str
            The name of the explicit Runge-Kutta method
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
        # Verify name is a string
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, not {type(name)}")

        # Verify A, b, and c are np.ndarrays
        for arr in zip(("A", "b", "c"), (A, b, c)):
            if not isinstance(arr[1], np.ndarray):
                raise TypeError(f"{arr[0]} must be a numpy.ndarray, "
                                f"not {type(arr[1])}")

        # Verify A is a square matrix
        if (A.shape[0] != A.shape[1]) or (len(A.shape) > 2):
            raise ValueError("Invalid dimensions: A must have dimensions "
                             f"(n, n), but instead has dimensions {A.shape}")

        # Verify b and c are vectors
        for arr in zip(("b", "c"), (b, c)):
            if len(arr[1].shape) > 1:
                raise ValueError(f"Invalid dimensions: {arr[0]} must have "
                                 "dimensions (n,), but instead has dimensions "
                                 f"{arr[1].shape}")

        # Verify the dimensions of A, b, and c match
        if not A.shape[0] == b.size == c.size:
            raise ValueError("Invalid dimensions: dim(A), dim(b), and dim(c) "
                             "must be (n,n), n, and n, but instead dim(A) = "
                             f"{A.shape}, dim(b) = {b.size}, and dim(c) = "
                             f"{c.size}")

        # Verify the elements of b sum to 1
        if not isclose(np.sum(b), 1.0, abs_tol=1e-8):
            raise ValueError("Invalid weights: the elements of b must sum "
                             f"to 1, but instead sum to {np.sum(b)}")

        # Verify the first element of c is 0
        if c[0] != 0.0:
            raise ValueError("Invalid nodes: c[0] must be 0 for the "
                             "Runge-Kutta method to be explicit, but instead "
                             f"is {c[0]}")

        # Verify the first row of A is only zeros
        A0 = A[0,:]
        if A0[A0 != 0].size > 0:
            raise ValueError("Invalid matrix: the first row of A must be "
                             "all 0s for the Runge-Kutta method to be "
                             f"explicit, but instead is {A0}")

        # Verify the rows of A sum to the elements of c
        bad_idxs = np.array([i for i in range(c.size)
                            if not isclose(np.sum(A[i,:]), c[i], abs_tol=1e-8)])
        bad_Ais = np.array([np.sum(A[i,:]) for i in bad_idxs])
        bad_cs = np.array([c[i] for i in bad_idxs])
        if bad_idxs.size > 0:
            raise ValueError("Invalid matrix or nodes: the rows of A must sum "
                             f"to the elements of c, but row(s) {bad_idxs} "
                             f"of A sum to {bad_Ais}, while element(s) "
                             f"{bad_idxs} of c are {bad_cs}")
        
        # Initialize class attributes
        self.name = name
        self.A = A
        self.b = b
        self.c = c
        self.n_stages = b.size


    def __str__(self) -> None:
        """
        Provides a printable representation of the ExplicitRK class.

        Returns
        -------
        None
        """
        rep_str = f"{self.name}\n"
        rep_str += self.tableau()
        return rep_str


    def __eq__(self, other: Any) -> bool:
        """
        Determines whether an instance of the ExplicitRK class is equal
        to another object.

        Parameters
        ----------
        other : Any
            Object being compared to an instance of the ExplicitRK class.

        Returns
        -------
        bool
            True if other is equal to self, False otherwise.
        """
        class_match = isinstance(other, ExplicitRK)
        param_match = (self.A == other.A) and (self.b == other.b) \
                      and (self.c == other.c)
        return class_match and param_match


    def tableau(self, output: str = "") -> str:
        """
        Determines the Butcher tableau of the explicit Runge-Kutta method.

        Parameters
        ----------
        output : str, optional
            Path and file name to where the Butcher tableau will be saved.
            Empty by default. If left unspecified, the tableau is not saved.

        Returns
        -------
        str
            String representing the Butcher tableau.
        """
        # Control spacing in tableau
        spaces = np.zeros(self.n_stages + 1, dtype=int)
        spaces[0] = max([len(str(ci)) for ci in self.c])

        for i in range(self.n_stages):
            spaces[i+1] = max([len(str(aij)) for aij in self.A[:,i]] \
                        + [len(str(self.b[i]))])
            
        # Initialize tableau string and add c_i rows
        tableau = ""
        for i in range(self.n_stages):
            row = ""
            row += f"{self.c[i]}" + " "*(spaces[0] - len(str(self.c[i]))) + " |"
            for j in range(self.n_stages):
                row += f" {self.A[i,j]}" + " "*(spaces[j+1] - len(str(self.A[i,j])))
            row += "\n"
            tableau += row
            
            if i == self.n_stages - 1:
                tableau += "-"*(len(row) - 1)
                tableau += "\n"

        # Add b_i rows  
        tableau += " "*spaces[0] + " |"
        for i in range(self.n_stages):
            tableau += f" {self.b[i]}" + " "*(spaces[i+1] - len(str(self.b[i])))

        # Save result
        if bool(output):
            with open(output, "w") as f:
                f.write(f"Butcher tableau for {self.name}:\n")
                f.write(tableau)
            f.close()

        return tableau


    def _step(self, f: Callable[[float, Union[float, np.ndarray]], np.ndarray],
              tn: float, yn: np.ndarray, h: float,
              args: tuple[Any]) -> np.ndarray:
        """
        Advances the numerical solution by one time step and updates
        solution value.

        Parameters
        ----------
        f : Callable[[float, Union[float, numpy.ndarray]], numpy.ndarray]
            The function on the right-hand side of the ODE y' = f(t, y).
        tn : float
            The current time value.
        yn : numpy.ndarray
            The current numerical solution value.
        h : float
            The time step.
        args : tuple[Any]
            Extra arguments to pass to f (i.e. model parameters).
        
        Returns
        -------
        numpy.ndarray
            The updated numerical solution value.
        """
        # Compute k values
        k = np.zeros((self.n_stages, yn.size))
        k[0] = f(tn, yn, *args)
        for i in range(1, self.n_stages):
            k[i] = f(tn + h*self.c[i], yn + h*np.dot(self.A[i,:], k), *args)
        
        # Compute updated solution value
        yn1 = yn + h*np.dot(self.b, k)
        return yn1


    def solve(self, f: Callable[[float, Union[float, np.ndarray]], np.ndarray],
              y0: Union[float, np.ndarray], t0: float, tf: float, h: float,
              args: tuple[Any] = (), verbose: bool = True,
              output: str = "") -> tuple[np.ndarray, np.ndarray]:
        """
        Numerically solves the ODE y' = f(t, y) with initial condition
        y(t0) = y0 on the interval [t0, tf] with time step h.

        Parameters
        ----------
        f : Callable[[float, Union[float, numpy.ndarray]], numpy.ndarray]
            The function on the right-hand side of the ODE y' = f(t, y).
        y0 : Union[float, numpy.ndarray]
            The initial condition y(t0) = y0.
        t0 : float
            The initial/start time.
        tf : float
            The final/end time.
        h : float
            The (constant) time step size.
        args : tuple[Any], optional
            Extra arguments to pass to f (i.e. model parameters).
            Empty by default.
        verbose : bool, optional
            If True, provides a progress bar indicating the solution progress
            and a performance report once the solution is calculated. If False,
            nothing is displayed. Set to True by default.
        output : str, optional
            Path and file name to where the results will be saved.
            Empty by default. If left unspecified, the results are not saved.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The time values and numerical solution values.
        """
        # Set initial condition to array if initial condition is a scalar
        is_scalar = isinstance(y0, int) or isinstance(y0, float)
        if is_scalar:
            y0 = np.array([y0])
        
        # Initialize time values and empty solution array
        t = np.arange(t0, tf + h, h)
        y = np.zeros((t.size, y0.size))

        # Save time at start of solution
        if verbose:
            print('Solving ODE...')
            start_time = time()

        # Set initial solution value to initial conditions
        y[0,:] = y0

        # Solve ODE by looping over _step function
        if verbose:
            for i in tqdm(range(t.size - 1)):
                y[i+1,:] = self._step(f, t[i], y[i,:], h, args=args)

        else:
            for i in range(t.size - 1):
                y[i+1,:] = self._step(f, t[i], y[i,:], h, args=args)
        
        # Calculate solution time
        if verbose:
            end_time = time() - start_time
            print(f"Solution complete! Time elapsed: {int(end_time//60)} "
                f"min {round(end_time % 60, 3)} sec.")

        # Save results
        if bool(output):
            sol = np.zeros((y.shape[0], y.shape[1] + 1))
            sol[:,0] = t
            sol[:,1:] = y
            np.savetxt(output, sol)

        # Set solution to a 1D array for scalar equations
        if is_scalar:
            y = y[:,0]

        return t, y