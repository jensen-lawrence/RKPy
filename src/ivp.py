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
    Docstring
    """

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------

    def __init__(self, f: Callable, y0: Union[float, np.ndarray],
                 t: tuple[float], dt: float, solver: RungeKutta,
                 args: tuple[Any] = (), rtol: float = None, atol: float = None,
                 dtmin: float = None, dtmax: float = None,
                 output: str = None) -> None:
        self.f = f
        self.y0 = y0
        self.t = t
        self.dt = dt
        self.solver = solver
        self.args = args
        self.rtol = rtol if rtol is not None else 1e-8
        self.atol = atol if atol is not None else 1e-8
        self.dtmin = dtmin if dtmin is not None else 0.01*dt
        self.dtmax = dtmax if dtmax is not None else 100.0*dt
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

        if self.solver.isadaptive:
            rep_str += f"- Minimum time step: {self.dtmin}\n"
            rep_str += f"- Maximum time step: {self.dtmax}\n"
            rep_str += f"- Relative error tolerance: {self.rtol}\n"
            rep_str += f"- Absolute error tolerance: {self.atol}\n"

        rep_str += f"- Solution method: {self.solver}"
        return rep_str


    def __eq__(self, other) -> bool:
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
            raise ValueError("t must have 2 elements, but instead has "
                             f"{len(self.t)} element(s)")

        for ti in self.t:
            if not (isinstance(ti, int) or isinstance(ti, float)):
                raise TypeError("Elements of t must be real numbers, not "
                                f"{type(self.t[0]), type(self.t[1])}")

        if self.t[1] < self.t[0]:
            raise ValueError("The second element of t must be equal to or "
                             "greater than the first element of t")

        if not isinstance(self.solver, RungeKutta):
            raise TypeError("solver must be an instance of RungeKutta, "
                            f"not {type(self.solver)}")

        if not (isinstance(self.args, list) or isinstance(self.args, tuple) or
                isinstance(self.args, np.ndarray)):
            self.args = (self.args,)

        if not isinstance(self.dt, float):
            raise TypeError(f"dt must be a float, not {type(self.dt)}")
        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        if self.solver.isadaptive:
            for val in zip(("dtmin", "dtmax", "rtol", "atol"),
                           (self.dtmin, self.dtmax, self.rtol, self.atol)):
                if not isinstance(val[1], float):
                    raise TypeError(f"{val[0]} must be a float, not "
                                    f"{type(val[0])}")
                if val[1] <= 0.0:
                    raise ValueError(f"{val[0]} must be greater than zero")

        if self.output is not None:
            if not isinstance(self.output, str):
                raise TypeError("output must be a str, not "
                                f"{type(self.output)}")

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    def solve(self, output: str = None) -> tuple[np.ndarray]:
        
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

        isscalar = isinstance(y0, int) or isinstance(y0, float)
        if isscalar:
            y0 = np.array([y0])

        if not isadaptive:
            b = self.solver.b
            t = np.arange(t0, tf + dt, dt)
            y = np.zeros((t.size, y0.size))
            y[0,:] = y0

            if isexplicit:
                for i in range(t.size - 1):
                    y[i+1,:] = _explicit_fixed_step(f, t[i], y[i,:], dt, A,
                                                    b, c, n_stages, args)

            else:
                for i in range(t.size - 1):
                    y[i+1,:] = _implicit_fixed_step(f, t[i], y[i,:], dt, A,
                                                    b, c, n_stages, args, rtol)

        else:
            b1 = self.solver.b1
            b2 = self.solver.b2
            t = [t0]
            y = [y0]
            ti = t0
            yi = y0
            m = 1.0/np.sqrt(y0.size)

            if isexplicit:
                while ti < tf:
                    dt = min(dt, tf - ti)
                    ti += dt
                    dt, yi = _explicit_adaptive_step(f, ti, yi, dt, A, b1, b2,
                                                     c, n_stages, args, dtmin,
                                                     dtmax, rtol, atol, m)
                    t.append(ti)
                    y.append(yi)
                    
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

        if output is not None:
            sol = np.zeros((y.shape[0], y.shape[1] + 1))
            sol[:,0] = t
            sol[:,1:] = y
            np.savetxt(output, sol)

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

def _explicit_fixed_step(f, tn, yn, dt, A, b, c, n_stages, args):
    # Compute k values
    k = np.zeros((n_stages, yn.size))
    k[0] = f(tn, yn, *args)
    for i in range(1, n_stages):
        k[i] = f(tn + dt*c[i], yn + dt*np.dot(A[i,:], k), *args)
    
    # Compute updated solution value
    yn1 = yn + dt*np.dot(b, k)

    return yn1


def _implicit_fixed_step(f, tn, yn, dt, A, b, c, n_stages, args):
    # Compute k values
    k = np.zeros((n_stages, yn.size))
    k0 = np.zeros(n_stages)
    one = np.ones(n_stages)
    for i in range(yn.size):
        k[:,i] = fsolve(lambda x: x - f(tn*one + dt*c, yn[i]*one \
                        + dt*np.dot(A, x), *args), k0, xtol=1e-9)

    # Compute updated solution value
    yn1 = yn + dt*np.dot(b, k)

    return yn1


def _explicit_adaptive_step(f, tn, yn, dt, A, b1, b2, c, n_stages, args, dtmin, dtmax, rtol, atol, m):
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


def _implicit_adaptive_step(f, tn, yn, dt, A, b1, b2, c, n_stages, args, dtmin, dtmax, rtol, atol, m):
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
