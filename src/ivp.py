# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from runge_kutta import RungeKutta
from typing import Any, Callable, Union

# ------------------------------------------------------------------------------
# InitialValueProblem Class
# ------------------------------------------------------------------------------

class InitialValueProblem:
    """
    Docstring
    """

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------

    def __init__(self, f: Callable, y0: Union[float, np.ndarray],
                 t: tuple[float], dt: float, solver: RungeKutta,
                 args: tuple[Any] = (), rtol: float = None, atol: float = None,
                 output: str = None) -> None:
        self.f = f
        self.y0 = y0
        self.t = t
        self.dt = dt
        self.solver = solver
        self.args = args
        self.rtol = rtol if rtol is not None else 1e-9
        self.atol = atol if atol is not None else 1e-9
        self.output = output
        self._check_parameters()


    def __str__(self) -> None:
        rep_str = "Initial value problem\n"
        rep_str += f"- Function: {self.f}\n"
        
        if self.args != ():
            rep_str += f"- Additional arguments: {self.args}\n"

        rep_str += f"- Initial conditions: {self.y0}\n"
        rep_str += f"- Time range: {self.t}\n"
        rep_str += f"- Time step: {self.dt}\n"
        rep_str += f"- Relative tolerance: {self.rtol}\n"
        rep_str += f"- Absolute tolerance: {self.atol}\n"
        rep_str += f"- Solution method: {self.solver}"
        return rep_str


    def __eq__(self, other) -> bool:
        if isinstance(other, InitialValueProblem):
            param_match = self.f == other.f
            param_match &= np.array_equal(self.y0, other.y0)
            param_match &= self.t == other.t
            param_match &= self.dt == other.dt
            param_match &= self.solver == other.solver
            param_match &= self.args == other.args
            param_match &= self.rtol == other.rtol
            param_match &= self.atol == other.atol
            return param_match

        return False


    def _check_parameters(self) -> None:
        if not isinstance(self.f, Callable):
            raise TypeError(f"f must be a function, not {type(self.f)}")

        if not (isinstance(self.y0, int) or isinstance(self.y0, float) \
                or isinstance(self.y0, np.ndarray)):
            raise TypeError("y0 must be a real number or numpy.ndarray of "
                            f"real numbers, not {type(self.y0)}")

        if not (isinstance(self.t, list) or isinstance(self.t, tuple) or
                isinstance(self.t, np.ndarray)):
            raise TypeError("t must be an array-like object, not"
                            f"{type(self.f)}")

        if len(self.t) != 2:
            raise ValueError("t must have length 2, but instead has length "
                             f"{len(self.t)}")

        for ti in self.t:
            if not (isinstance(ti, int) or isinstance(ti, float)):
                raise TypeError("Elements of t must be real numbers, not "
                                f"({type(self.t[0]), type(self.t[1])})")

        if self.t[1] < self.t[0]:
            raise ValueError("The second element of t must be equal to or "
                             "greater than the first element of t")

        if not isinstance(self.solver, RungeKutta):
            raise TypeError("solver must be an instance of RungeKutta, "
                            f"not {type(self.solver)}")

        if not (isinstance(self.args, list) or isinstance(self.args, tuple) or
                isinstance(self.args, np.ndarray)):
            self.args = (self.args,)

        for val in zip(("dt", "rtol", "atol"), (self.dt, self.rtol, self.atol)):
            if not isinstance(val[1], float):
                raise TypeError(f"{val[0]} must be a float, not {type(val[0])}")
            if val[1] <= 0.0:
                raise ValueError(f"{val[0]} must be greater than zero")

        if self.output is not None:
            if not isinstance(self.output, str):
                raise TypeError("output must be a str, not "
                                f"{type(self.output)}")

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    def solve(self) -> tuple[np.ndarray]:
        pass


# ------------------------------------------------------------------------------
# IVP Function
# ------------------------------------------------------------------------------

def IVP(f: Callable, y0: Union[float, np.ndarray], t: tuple[float], dt: float,
        solver: RungeKutta, args: tuple[Any] = (), rtol: float = None,
        atol: float = None, output: str = None) -> InitialValueProblem:
    """
    Wrapper function for the InitialValueProblem class to provide IVP as an
    abbreviation of InitialValueProblem.
    """
    return InitialValueProblem(f, y0, t, dt, solver,
                               args=args, rtol=rtol, atol=atol, output=output)

#     def _step(self, f: Callable[[float, Union[float, np.ndarray]], np.ndarray],
#               tn: float, yn: np.ndarray, h: float,
#               args: tuple[Any]) -> np.ndarray:
#         """
#         Advances the numerical solution by one time step and updates
#         solution value.

#         Parameters
#         ----------
#         f : Callable[[float, Union[float, numpy.ndarray]], numpy.ndarray]
#             The function on the right-hand side of the ODE y' = f(t, y).
#         tn : float
#             The current time value.
#         yn : numpy.ndarray
#             The current numerical solution value.
#         h : float
#             The time step.
#         args : tuple[Any]
#             Extra arguments to pass to f (i.e. model parameters).
        
#         Returns
#         -------
#         numpy.ndarray
#             The updated numerical solution value.
#         """
#         # Compute k values
#         k = np.zeros((self.n_stages, yn.size))
#         k[0] = f(tn, yn, *args)
#         for i in range(1, self.n_stages):
#             k[i] = f(tn + h*self.c[i], yn + h*np.dot(self.A[i,:], k), *args)
        
#         # Compute updated solution value
#         yn1 = yn + h*np.dot(self.b, k)
#         return yn1


#     def solve(self, f: Callable[[float, Union[float, np.ndarray]], np.ndarray],
#               y0: Union[float, np.ndarray], t0: float, tf: float, h: float,
#               args: tuple[Any] = (), verbose: bool = True,
#               output: str = "") -> tuple[np.ndarray, np.ndarray]:
#         """
#         Numerically solves the ODE y' = f(t, y) with initial condition
#         y(t0) = y0 on the interval [t0, tf] with time step h.

#         Parameters
#         ----------
#         f : Callable[[float, Union[float, numpy.ndarray]], numpy.ndarray]
#             The function on the right-hand side of the ODE y' = f(t, y).
#         y0 : Union[float, numpy.ndarray]
#             The initial condition y(t0) = y0.
#         t0 : float
#             The initial/start time.
#         tf : float
#             The final/end time.
#         h : float
#             The (constant) time step size.
#         args : tuple[Any], optional
#             Extra arguments to pass to f (i.e. model parameters).
#             Empty by default.
#         verbose : bool, optional
#             If True, provides a progress bar indicating the solution progress
#             and a performance report once the solution is calculated. If False,
#             nothing is displayed. Set to True by default.
#         output : str, optional
#             Path and file name to where the results will be saved.
#             Empty by default. If left unspecified, the results are not saved.

#         Returns
#         -------
#         tuple[numpy.ndarray, numpy.ndarray]
#             The time values and numerical solution values.
#         """
#         # Set initial condition to array if initial condition is a scalar
#         is_scalar = isinstance(y0, int) or isinstance(y0, float)
#         if is_scalar:
#             y0 = np.array([y0])
        
#         # Initialize time values and empty solution array
#         t = np.arange(t0, tf + h, h)
#         y = np.zeros((t.size, y0.size))

#         # Save time at start of solution
#         if verbose:
#             print('Solving ODE...')
#             start_time = time()

#         # Set initial solution value to initial conditions
#         y[0,:] = y0

#         # Solve ODE by looping over _step function
#         if verbose:
#             for i in tqdm(range(t.size - 1)):
#                 y[i+1,:] = self._step(f, t[i], y[i,:], h, args=args)

#         else:
#             for i in range(t.size - 1):
#                 y[i+1,:] = self._step(f, t[i], y[i,:], h, args=args)
        
#         # Calculate solution time
#         if verbose:
#             end_time = time() - start_time
#             print(f"Solution complete! Time elapsed: {int(end_time//60)} "
#                 f"min {round(end_time % 60, 3)} sec.")

#         # Save results
#         if bool(output):
#             sol = np.zeros((y.shape[0], y.shape[1] + 1))
#             sol[:,0] = t
#             sol[:,1:] = y
#             np.savetxt(output, sol)

#         # Set solution to a 1D array for scalar equations
#         if is_scalar:
#             y = y[:,0]

#         return t, y