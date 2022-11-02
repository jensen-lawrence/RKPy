# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from runge_kutta import RungeKutta
from typing import Any, Callable, Union
from scipy.optimize import fsolve

# ------------------------------------------------------------------------------
# InitialValueProblem Class
# ------------------------------------------------------------------------------

class InitialValueProblem:
    """
    Class for initial value problems of the form

    \\begin{equation}
        y'(t) = f(t, y(t)), y(t_0) = y_0, t \in [t_0, t_f]
    \end{equation}

    An initial value problem is characterized by f, y0, t, dt,
    and solver, where:
    - f is the right-hand side of the IVP.
    - y0 is the initial condition, i.e. the value of the solution to the IVP
    at the initial time.
    - t = (t0, tf) is the range of times over which the IVP is to be solved.
    - dt is the time step used when solving the IVP. For fixed step Runge-Kutta
    methods, dt remains constant, but for adaptive Runge-Kutta methods, it is
    updated each step based on the error in the solution.
    - solver is the Runge-Kutta method used to solve the IVP.

    The behaviour or performance of the class can be refined using some
    optional keyword arguments:
    - args is a tuple (p1, p2, ...) containing any additional arguments to
    be passed to f for IVPs of the form y'(t) = f(t, y(t), p1, p2, ...).
    - rtol is the relative error tolerance used when updating the step size
    in adaptive methods.
    - atol is the absolute error tolerance used when updating the step size
    in adaptive methods.
    - dtmin is the minimum allowed step size for adaptive methods.
    - dtmax is the maximum allowed step size for adaptive methods.
    - output is a string containing the path to where the solution results
    will be saved.
    """

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------

    def __init__(self, f: Callable, y0: Union[float, np.ndarray],
                 t: tuple[float], dt: float, solver: RungeKutta,
                 args: tuple[Any] = (), rtol: float = None, atol: float = None,
                 dtmin: float = None, dtmax: float = None,
                 output: str = None) -> None:
        """
        Initializes the InitialValueProblem class.

        Parameters
        ----------
        f : Callable
            The function on the right-hand side of the IVP.
        y0 : Union[float, numpy.ndarray]
            The initial condition/solution to the IVP at the initial time.
        t : tuple[float]
            The tuple (t0, tf), t0 < tf, where t0 is the time at which the
            solution to the IVP starts and tf is the time at which the
            solution to the IVP ends.
        dt : float
            The time step used in the Runge-Kutta method that solves the IVP.
            For fixed step methods, dt is constant, but for adaptive methods,
            dt is updated according to the error at each step.
        solver : RungeKutta
            The Runge-Kutta method used to solve the IVP. For more information,
            see the RungeKutta class in runge_kutta.py
        args : tuple[Any], optional
            Additional arguments passed to f if required. () by default.
        rtol : float, optional
            The relative error tolerance used when calculating the updated
            step size in adaptive methods. Set to 1e-9 if left unspecified.
        atol : float, optional
            The absolute error tolerance used when calculating the updated
            step size in adaptive methods. Set to 1e-9 if left unspecified.
        dtmin : float, optional
            The minimum allowed step size for adaptive methods. Set to 0.01*dt
            (where dt is the initial time step) if left unspecified.
        dtmax : float, optional
            The maximum allowed step size for adaptive methods. Set to 100.0*dt
            (where dt is the initial time step) if left unspecified.
        output : str, optional
            Path and file name to where the IVP solution will be saved.
            None by default. If left unspecified, the solution is not saved.

        Returns
        -------
        None
        """
        self.f = f
        self.y0 = y0
        self.t = t
        self.dt = dt
        self.solver = solver
        self.args = args
        self.rtol = rtol if rtol is not None else 1e-9
        self.atol = atol if atol is not None else 1e-9
        self.dtmin = dtmin if dtmin is not None else 0.01*dt
        self.dtmax = dtmax if dtmax is not None else 100.0*dt
        self.output = output
        self._check_parameters()


    def __str__(self) -> str:
        """
        Provides a printable representation of the InitialValueProblem class.

        Returns
        -------
        None
        """
        rep_str = "Initial value problem\n"
        rep_str += f"- Function: {self.f}\n"
        
        if self.args != ():
            rep_str += f"- Additional arguments: {self.args}\n"

        rep_str += f"- Initial conditions: {self.y0}\n"
        rep_str += f"- Time range: {self.t}\n"
        rep_str += f"- Time step: {self.dt}\n"

        if self.solver.isadaptive:
            rep_str += f"- Minimum time step: {self.dtmin}\n"
            rep_str += f"- Maximum time step: {self.dtmax}\n"
            rep_str += f"- Relative error tolerance: {self.rtol}\n"
            rep_str += f"- Absolute error tolerance: {self.atol}\n"

        rep_str += f"- Solution method: {self.solver}"
        return rep_str


    def __eq__(self, other) -> bool:
        """
        Determines whether an instance of the InitialValueProblem class
        is equal to another object.

        Parameters
        ----------
        other : Any
            Object being compared to an instance of the InitialValueProblem
            class.

        Returns
        -------
        bool
            True if other is equal to self, False otherwise.
        """
        if isinstance(other, InitialValueProblem):
            param_match = self.f == other.f
            param_match &= np.array_equal(self.y0, other.y0)
            param_match &= self.t == other.t
            param_match &= self.dt == other.dt
            param_match &= self.dtmin == other.dtmin
            param_match &= self.dtmax == other.dtmax
            param_match &= self.solver == other.solver
            param_match &= self.args == other.args
            param_match &= self.rtol == other.rtol
            param_match &= self.atol == other.atol
            return param_match

        return False


    def _check_parameters(self) -> None:
        """
        Determines whether the parameters provided to the InitialValueProblem
        class are valid parameters.

        Returns
        -------
        None

        Raises
        ------
        TypeError
        - f is not a function
        - y0 is not an int, float, or numpy.ndarray
        - t is not a list, tuple, or numpy.ndarray
        - The elements of t are not ints or floats
        - solver is not an instance of RungeKutta
        - dt is not a float
        - dtmin and dtmax are not floats (adaptive methods)
        - rtol and atol are not floats (adaptive methods)
        - output is not a str

        ValueError
        - t does not have two elements
        - The second element of t is not greater than the first element of t
        - dt is not greater than zero
        - dtmin and dtmax are not greater than zero (adaptive methods)
        - rtol and atol are not greater than zero (adaptive methods)
        """
        # Verify f is a function
        if not isinstance(self.f, Callable):
            raise TypeError(f"f must be a function, not {type(self.f)}")

        # Verify y0 is a real number or np.array
        if not (isinstance(self.y0, int) or isinstance(self.y0, float) \
                or isinstance(self.y0, np.ndarray)):
            raise TypeError("y0 must be a real number or numpy.ndarray of "
                            f"real numbers, not {type(self.y0)}")

        # Verify t is an array-like object
        if not (isinstance(self.t, list) or isinstance(self.t, tuple) or
                isinstance(self.t, np.ndarray)):
            raise TypeError("t must be an array-like object, not"
                            f"{type(self.f)}")

        # Verify t has two elements
        if len(self.t) != 2:
            raise ValueError("t must have 2 elements, but instead has "
                             f"{len(self.t)} element(s)")

        # Verify the elements of t are real numbers
        for ti in self.t:
            if not (isinstance(ti, int) or isinstance(ti, float)):
                raise TypeError("Elements of t must be real numbers, not "
                                f"{type(self.t[0]), type(self.t[1])}")

        # Verify the final time is greater than the initial time
        if self.t[1] < self.t[0]:
            raise ValueError("The second element of t must be equal to or "
                             "greater than the first element of t")

        # Verify solver is an instance of the RungeKutta class
        if not isinstance(self.solver, RungeKutta):
            raise TypeError("solver must be an instance of RungeKutta, "
                            f"not {type(self.solver)}")

        # Verify args is an array-like object
        if not (isinstance(self.args, list) or isinstance(self.args, tuple) or
                isinstance(self.args, np.ndarray)):
            self.args = (self.args,)

        # Verify dt is a float
        if not isinstance(self.dt, float):
            raise TypeError(f"dt must be a float, not {type(self.dt)}")

        # Verify dt is greater than zero
        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        # Additional checks for adaptive methods
        if self.solver.isadaptive:
            for val in zip(("dtmin", "dtmax", "rtol", "atol"),
                           (self.dtmin, self.dtmax, self.rtol, self.atol)):
                # Verify dtmin, dtmax, rtol, and atol are floats
                if not isinstance(val[1], float):
                    raise TypeError(f"{val[0]} must be a float, not "
                                    f"{type(val[0])}")

                # Verify dtmin, dtmax, rtol, and atol are greater than zero
                if val[1] <= 0.0:
                    raise ValueError(f"{val[0]} must be greater than zero")

        # Verify output is a string
        if self.output is not None:
            if not isinstance(self.output, str):
                raise TypeError("output must be a str, not "
                                f"{type(self.output)}")

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    def solve(self, output: str = None) -> tuple[np.ndarray]:
        """
        Solves the initial value problem described by the instance of
        InitialValueProblem.

        Parameters
        ----------
        output : str, optional
            Path and file name to where the IVP solution will be saved.
            None by default. If left unspecified, the solution is not saved.

        Returns
        -------
        tuple[numpy.ndarray]
            Tuple containing the array of time values and the array of
            solution values.
        """
        
        # Initialize parameters from IVP and solver
        f = self.f
        y0 = self.y0
        t0, tf = self.t
        dt = self.dt
        args = self.args
        dtmin = self.dtmin
        dtmax = self.dtmax
        rtol = self.rtol
        atol = self.atol
        A = self.solver.A
        c = self.solver.c
        n_stages = self.solver.n_stages
        isexplicit = self.solver.isexplicit
        isadaptive = self.solver.isadaptive

        # Check if the ODE is a scalar equation
        isscalar = isinstance(y0, int) or isinstance(y0, float)
        if isscalar:
            y0 = np.array([y0])

        # Solver loops for fixed step methods
        if not isadaptive:
            b = self.solver.b
            t = np.arange(t0, tf + dt, dt)
            y = np.zeros((t.size, y0.size))
            y[0,:] = y0

            # Explicit fixed step solver
            if isexplicit:
                for i in range(t.size - 1):
                    y[i+1,:] = _explicit_fixed_step(f, t[i], y[i,:], dt, A,
                                                    b, c, n_stages, args)

            # Implicit fixed step solver
            else:
                for i in range(t.size - 1):
                    y[i+1,:] = _implicit_fixed_step(f, t[i], y[i,:], dt, A,
                                                    b, c, n_stages, args, rtol)

        # Solver loops for adaptive methods
        else:
            b1 = self.solver.b1
            b2 = self.solver.b2
            t = [t0]
            y = [y0]
            ti = t0
            yi = y0
            m = 1.0/np.sqrt(y0.size)

            # Explicit adaptive solver
            if isexplicit:
                while ti < tf:
                    dt = min(dt, tf - ti)
                    ti += dt
                    dt, yi = _explicit_adaptive_step(f, ti, yi, dt, A, b1, b2,
                                                     c, n_stages, args, dtmin,
                                                     dtmax, rtol, atol, m)
                    t.append(ti)
                    y.append(yi)

            # Implicit adaptive solver
            else:
                while ti <= tf:
                    dt = min(dt, tf - ti)
                    ti += dt
                    dt, yi = _implicit_adaptive_step(f, ti, yi, dt, A, b1, b2,
                                                     c, n_stages, args, dtmin,
                                                     dtmax, rtol, atol, m)
                    t.append(ti)
                    y.append(yi)

            t = np.array(t)
            y = np.array(y)    

        # Output solution
        if output is not None:
            sol = np.zeros((y.shape[0], y.shape[1] + 1))
            sol[:,0] = t
            sol[:,1:] = y
            np.savetxt(output, sol)

        # Format solution array for scalar equations
        if isscalar:
            y = y[:,0]

        return t, y

# ------------------------------------------------------------------------------
# IVP Function
# ------------------------------------------------------------------------------

def IVP(f: Callable, y0: Union[float, np.ndarray], t: tuple[float], dt: float,
        solver: RungeKutta, args: tuple[Any] = (), rtol: float = None,
        output: str = None) -> InitialValueProblem:
    """
    Wrapper function for the InitialValueProblem class to provide IVP as an
    abbreviation of InitialValueProblem.
    """
    return InitialValueProblem(f, y0, t, dt, solver,
                               args=args, rtol=rtol, output=output)

# ------------------------------------------------------------------------------
# Step Functions
# ------------------------------------------------------------------------------

def _explicit_fixed_step(f: Callable, tn: float, yn: np.ndarray, dt: float,
                         A: np.ndarray, b: np.ndarray, c: np.ndarray,
                         n_stages: int, args: tuple[Any]) -> np.ndarray:
    """
    Advances the solution to an initial value problem by one time step
    using an explicit Runge-Kutta method with a fixed time step.

    Parameters
    ----------
    f : Callable
        The function on the right-hand side of the IVP.
    tn : float
        The current time value.
    yn : numpy.ndarray
        The current value of the solution to the IVP.
    dt : float
        The time step used by the Runge-Kutta method.
    A : numpy.ndarray
        The Runge-Kutta matrix.
    b : numpy.ndarray
        The vector of Runge-Kutta weights.
    c : numpy.ndarray
        The vector of Runge-Kutta nodes.
    n_stages : int
        The number of stages the Runge-Kutta method has.
    args : tuple[Any]
        Additional arguments passed to f if required.

    Returns
    -------
    numpy.ndarray
        Updated solution to the initial value problem.
    """
    # Compute k values
    k = np.zeros((n_stages, yn.size))
    k[0] = f(tn, yn, *args)
    for i in range(1, n_stages):
        k[i] = f(tn + dt*c[i], yn + dt*np.dot(A[i,:], k), *args)
    
    # Compute solution
    yn1 = yn + dt*np.dot(b, k)

    return yn1


def _implicit_fixed_step(f: Callable, tn: float, yn: np.ndarray, dt: float,
                         A: np.ndarray, b: np.ndarray, c: np.ndarray,
                         n_stages: int, args: tuple[Any]) -> np.ndarray:
    """
    Advances the solution to an initial value problem by one time step
    using an implicit Runge-Kutta method with a fixed time step.

    Parameters
    ----------
    f : Callable
        The function on the right-hand side of the IVP.
    tn : float
        The current time value.
    yn : numpy.ndarray
        The current value of the solution to the IVP.
    dt : float
        The time step used by the Runge-Kutta method.
    A : numpy.ndarray
        The Runge-Kutta matrix.
    b : numpy.ndarray
        The vector of Runge-Kutta weights.
    c : numpy.ndarray
        The vector of Runge-Kutta nodes.
    n_stages : int
        The number of stages the Runge-Kutta method has.
    args : tuple[Any]
        Additional arguments passed to f if required.

    Returns
    -------
    numpy.ndarray
        Updated solution to the initial value problem.
    """
    # Compute k values
    k = np.zeros((n_stages, yn.size))
    k0 = np.zeros(n_stages)
    one = np.ones(n_stages)
    for i in range(yn.size):
        k[:,i] = fsolve(lambda x: x - f(tn*one + dt*c, yn[i]*one \
                        + dt*np.dot(A, x), *args), k0, xtol=1e-9)

    # Compute solution
    yn1 = yn + dt*np.dot(b, k)

    return yn1


def _explicit_adaptive_step(f: Callable, tn: float, yn: np.ndarray, dt: float,
                            A: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                            c: np.ndarray, n_stages: int, args: tuple[Any],
                            dtmin: float, dtmax: float, rtol: float,
                            atol: float, m: float) -> tuple[float, np.ndarray]:
    """
    Advances the solution to an initial value problem by one time step
    using an explicit Runge-Kutta method with an adaptive time step. Updates
    the time step according to the RK method's adaptive scheme.

    Parameters
    ----------
    f : Callable
        The function on the right-hand side of the IVP.
    tn : float
        The current time value.
    yn : numpy.ndarray
        The current value of the solution to the IVP.
    dt : float
        The time step used by the Runge-Kutta method.
    A : numpy.ndarray
        The Runge-Kutta matrix.
    b1 : numpy.ndarray
        The vector of Runge-Kutta weights for the primary method.
    b2 : numpy.ndarray
        The vector of Runge-Kutta weights for the embedded method.
    c : numpy.ndarray
        The vector of Runge-Kutta nodes.
    n_stages : int
        The number of stages the Runge-Kutta method has.
    args : tuple[Any]
        Additional arguments passed to f if required.
    dtmin : float, optional
        The minimum allowed step size.
    dtmax : float, optional
        The maximum allowed step size.
    rtol : float, optional
        The relative error tolerance used when calculating the updated
        step size.
    atol : float, optional
        The absolute error tolerance used when calculating the updated
        step size.
    m : float
        1/sqrt(n_stages)

    Returns
    -------
    tuple[float, numpy.ndarray]
        Updated time step and updated solution to the initial value problem.
    """
    # Compute k values
    k = np.zeros((n_stages, yn.size))
    k[0] = f(tn, yn, *args)
    for i in range(1, n_stages):
        k[i] = f(tn + dt*c[i], yn + dt*np.dot(A[i,:], k), *args)
    
    # Compute primary and embedded solutions
    yn1 = yn + dt*np.dot(b1, k)
    un1 = yn + dt*np.dot(b2, k)

    # Compute error and new step size
    err = np.abs(yn1 - un1)
    s = (m*np.linalg.norm(err/(atol + rtol*yn1)))**(-1/n_stages)
    dt1 = 0.8*dt*s
    dt1 = max(min(dt1, dtmax), dtmin)
        
    return dt1, yn1


def _implicit_adaptive_step(f: Callable, tn: float, yn: np.ndarray, dt: float,
                            A: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                            c: np.ndarray, n_stages: int, args: tuple[Any],
                            dtmin: float, dtmax: float, rtol: float,
                            atol: float, m: float) -> tuple[float, np.ndarray]:
    """
    Advances the solution to an initial value problem by one time step
    using an implicit Runge-Kutta method with an adaptive time step. Updates
    the time step according to the RK method's adaptive scheme.

    Parameters
    ----------
    f : Callable
        The function on the right-hand side of the IVP.
    tn : float
        The current time value.
    yn : numpy.ndarray
        The current value of the solution to the IVP.
    dt : float
        The time step used by the Runge-Kutta method.
    A : numpy.ndarray
        The Runge-Kutta matrix.
    b1 : numpy.ndarray
        The vector of Runge-Kutta weights for the primary method.
    b2 : numpy.ndarray
        The vector of Runge-Kutta weights for the embedded method.
    c : numpy.ndarray
        The vector of Runge-Kutta nodes.
    n_stages : int
        The number of stages the Runge-Kutta method has.
    args : tuple[Any]
        Additional arguments passed to f if required.
    dtmin : float, optional
        The minimum allowed step size.
    dtmax : float, optional
        The maximum allowed step size.
    rtol : float, optional
        The relative error tolerance used when calculating the updated
        step size.
    atol : float, optional
        The absolute error tolerance used when calculating the updated
        step size.
    m : float
        1/sqrt(n_stages)

    Returns
    -------
    tuple[float, numpy.ndarray]
        Updated time step and updated solution to the initial value problem.
    """
    # Compute k values
    k = np.zeros((n_stages, yn.size))
    k0 = np.zeros(n_stages)
    one = np.ones(n_stages)
    for i in range(yn.size):
        k[:,i] = fsolve(lambda x: x - f(tn*one + dt*c, yn[i]*one \
                        + dt*np.dot(A, x), *args), k0, xtol=1e-9)

    # Compute updated solution value
    yn1 = yn + dt*np.dot(b1, k)
    un1 = yn + dt*np.dot(b2, k)

    # Compute error and new step size
    err = np.abs(yn1 - un1)
    s = (m*np.linalg.norm(err/(atol + rtol*yn1)))**(-1/n_stages)
    dt1 = 0.8*dt*s
    dt1 = max(min(dt1, dtmax), dtmin)

    return dt1, yn1
