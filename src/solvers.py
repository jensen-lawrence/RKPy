# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from explicit_rk import ExplicitRK

# ------------------------------------------------------------------------------
# Solvers
# ------------------------------------------------------------------------------

# References:
# https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
# http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/Butcher.html

euler = ExplicitRK(
    name = "Euler's Method",
    A = np.array([[0.0]]),
    b = np.array([1.0]),
    c = np.array([0.0])
)

midpoint = ExplicitRK(
    name = "Midpoint Method",
    A = np.array([
        [0.0, 0.0],
        [0.5, 0.0]
    ]),
    b = np.array([0.0, 1.0]),
    c = np.array([0.0, 0.5])
)

heun2 = ExplicitRK(
    name = "Heun's Second-Order Method",
    A = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ]),
    b = np.array([0.5, 0.5]),
    c = np.array([0.0, 1.0])
)

ralston2 = ExplicitRK(
    name = "Ralston's Second-Order Method",
    A = np.array([
        [0.0, 0.0],
        [2.0/3.0, 0.0]
    ]),
    b = np.array([0.25, 0.75]),
    c = np.array([0.0, 2.0/3.0])
)

kutta3 = ExplicitRK(
    name = "Kutta's Third-Order Method",
    A = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [-1.0, 2.0, 0.0]
    ]),
    b = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0]),
    c = np.array([0.0, 0.5, 1.0])
)

heun3 = ExplicitRK(
    name = "Heun's Third-Order Method",
    A = np.array([
        [0.0,     0.0,     0.0],
        [1.0/3.0, 0.0,     0.0],
        [0.0,     2.0/3.0, 0.0]
    ]),
    b = np.array([0.25, 0.0, 0.75]),
    c = np.array([0.0, 1.0/3.0, 2.0/3.0])
)

ralston3 = ExplicitRK(
    name = "Ralston's Third-Order Method",
    A = np.array([
        [0.0, 0.0,  0.0],
        [0.5, 0.0,  0.0],
        [0.0, 0.75, 0.0]
    ]),
    b = np.array([2.0/9.0, 1.0/3.0, 4.0/9.0]),
    c = np.array([0.0, 0.5, 0.75])
)

wray3 = ExplicitRK(
    name = "Van der Houwen's/Wray's Third-Order Method",
    A = np.array([
        [0.0,      0.0,      0.0],
        [8.0/15.0, 0.0,      0.0],
        [0.25,     5.0/12.0, 0.0]
    ]),
    b = np.array([0.25, 0.0, 0.75]),
    c = np.array([0.0, 8.0/15.0, 2.0/3.0])
)

ssprk3 = ExplicitRK(
    name = "Third-Order Strong Stability Preserving Runge-Kutta Method",
    A = np.array([
        [0.0,  0.0,  0.0],
        [1.0,  0.0,  0.0],
        [0.25, 0.25, 0.0]
    ]),
    b = np.array([1.0/6.0, 1.0/6.0, 2.0/3.0]),
    c = np.array([0.0, 1.0, 0.5])
)

knoth_wolke = ExplicitRK(
    name = "Knoth-Wolke Method",
    A = np.array([
        [0.0,       0.0,       0.0],
        [1.0/3.0,   0.0,       0.0],
        [-3.0/16.0, 15.0/16.0, 0.0]
    ]),
    b = np.array([1.0/6.0, 3.0/10.0, 8.0/15.0]),
    c = np.array([0.0, 1.0/3.0, 0.75])
)

classic4 = ExplicitRK(
    name = "Classic Fourth-Order Method",
    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]),
    b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]),
    c = np.array([0.0, 0.5, 0.5, 1.0])
)

kutta3_8 = ExplicitRK(
    name = "3/8 Rule Fourth-Order Method",
    A = np.array([
        [0.0,      0.0,  0.0, 0.0],
        [1.0/3.0,  0.0,  0.0, 0.0],
        [-1.0/3.0, 1.0,  0.0, 0.0],
        [1.0,      -1.0, 1.0, 0.0]
    ]),
    b = np.array([0.125, 0.375, 0.375, 0.125]),
    c = np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
)

zonneveld = ExplicitRK(
    name = "Zonneveld's Fourth-Order Method",
    A = np.array([
        [0.0,      0.0,      0.0,       0.0,       0.0],
        [0.5,      0.0,      0.0,       0.0,       0.0],
        [0.0,      0.5,      0.0,       0.0,       0.0],
        [0.0,      0.0,      1.0,       0.0,       0.0],
        [5.0/32.0, 7.0/32.0, 13.0/32.0, -1.0/32.0, 0.0]
    ]),
    b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0, 0.0]),
    c = np.array([0.0, 0.5, 0.5, 1.0, 0.75])
)

cash_karp = ExplicitRK(
    name = "Cash-Karp Method",
    A = np.array([
        [0.0,            0.0,         0.0,           0.0,              0.0,          0.0],
        [0.2,            0.0,         0.0,           0.0,              0.0,          0.0],
        [0.075,          0.225,       0.0,           0.0,              0.0,          0.0],
        [0.3,            -0.9,        1.2,           0.0,              0.0,          0.0],
        [-11.0/54.0,     2.5,         -70.0/27.0,    35.0/27.0,        0.0,          0.0],
        [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, 0.0]
    ]),
    b = np.array([37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]),
    c = np.array([0.0, 0.2, 0.3, 0.6, 1.0, 0.875])
)

fehlberg5 = ExplicitRK(
    name = "Fehlberg's Fifth-Order Method",
    A = np.array([
        [0.0,           0.0,            0.0,            0.0,           0.0,        0.0],
        [0.25,          0.0,            0.0,            0.0,           0.0,        0.0],
        [3.0/32.0,      9.0/32.0,       0.0,            0.0,           0.0,        0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0,  0.0,           0.0,        0.0],
        [439.0/216.0,   -8.0,           3680.0/513.0,   -845.0/4104.0, 0.0,        0.0],
        [-8.0/27.0,     2.0,            -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0]
    ]),
    b = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0]),
    c = np.array([0.0, 0.25, 0.375, 12.0/13.0, 1.0, 0.5])
)

dormand_prince = ExplicitRK(
    name = "Dormand-Prince Method",
    A = np.array([
        [0.0,            0.0,             0.0,            0.0,          0.0,             0.0,       0.0],
        [0.2,            0.0,             0.0,            0.0,          0.0,             0.0,       0.0],
        [3.0/40.0,       9.0/40.0,        0.0,            0.0,          0.0,             0.0,       0.0],
        [44.0/45.0,      -56.0/15.0,      32.0/9.0,       0.0,          0.0,             0.0,       0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0,             0.0,       0.0],
        [9017.0/3168.0,  -355.0/33.0,     46732.0/5247.0, 49.0/176.0,   -5103.0/18656.0, 0.0,       0.0],
        [35.0/384.0,     0.0,             500.0/1113.0,   125.0/192.0,  -2187.0/6784.0,  11.0/84.0, 0.0]
    ]),
    b = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]),
    c = np.array([0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0])
)

verner = ExplicitRK(
    name = "Verner's Method",
    A = np.array([
        [0.0,             0.0,         0.0,              0.0,           0.0,             0.0, 0.0,            0.0],
        [1.0/6.0,         0.0,         0.0,              0.0,           0.0,             0.0, 0.0,            0.0],
        [4.0/75.0,        16.0/75.0,   0.0,              0.0,           0.0,             0.0, 0.0,            0.0],
        [5.0/6.0,         -8.0/3.0,    2.5,              0.0,           0.0,             0.0, 0.0,            0.0],
        [-165.0/64.0,     55.0/6.0,    -425.0/64.0,      85.0/96.0,     0.0,             0.0, 0.0,            0.0],
        [12.0/5.0,        -8.0,        4015.0/612.0,     -11.0/36.0,    88.0/255.0,      0.0, 0.0,            0.0],
        [-8263.0/15000.0, 124.0/75.0,  -643.0/680.0,     -81.0/250.0,   2484.0/10625.0,  0.0, 0.0,            0.0],
        [3501.0/1720.0,   -300.0/43.0, 297275.0/52632.0, -319.0/2322.0, 24068.0/84065.0, 0.0, 3850.0/26703.0, 0.0]
    ]),
    b = np.array([3.0/40.0, 0.0, 875.0/2244.0, 23.0/72.0, 264.0/1955.0, 0.0, 125.0/11592.0, 43.0/616.0]),
    c = np.array([0.0, 1.0/6.0, 4.0/15.0, 2.0/3.0, 5.0/6.0, 1.0, 1.0/15.0, 1.0])
)

fehlberg8 = ExplicitRK(
    name = "Fehlberg's Eighth-Order Method",
    A = np.array([
        [0.0,            0.0,      0.0,        0.0,          0.0,            0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [2.0/27.0,       0.0,      0.0,        0.0,          0.0,            0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [1.0/36.0,       1.0/12.0, 0.0,        0.0,          0.0,            0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [1.0/24.0,       0.0,      0.125,      0.0,          0.0,            0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [5.0/12.0,       0.0,      -25.0/16.0, 25.0/16.0,    0.0,            0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [0.05,           0.0,      0.0,        0.25,         0.2,            0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [-25.0/108.0,    0.0,      0.0,        125.0/108.0,  -65.0/27.0,     125.0/54.0,  0.0,           0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [31.0/300.0,     0.0,      0.0,        0.0,          61.0/225.0,     -2.0/9.0,    13.0/900.0,    0.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [2.0,            0.0,      0.0,        -53.0/6.0,    704.0/45.0,     -107.0/9.0,  67.0/90.0,     3.0,       0.0,        0.0,       0.0, 0.0, 0.0],
        [-91.0/108.0,    0.0,      0.0,        23.0/108.0,   -976.0/135.0,   311.0/54.0,  -19.0/60.0,    17.0/6.0,  -1.0/12.0,  0.0,       0.0, 0.0, 0.0],
        [2383.0/4100.0,  0.0,      0.0,        -341.0/164.0, 4496.0/1025.0,  -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0, 0.0, 0.0, 0.0],
        [3.0/205.0,      0.0,      0.0,        0.0,          0.0,            -6.0/41.0,   -3.0/205.0,    -3.0/41.0, 3.0/41.0,   6.0/41.0,  0.0, 0.0, 0.0],
        [-1777.0/4100.0, 0.0,      0.0,        -341.0/164.0, 4496.0/1025.0,  -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0, 0.0]
    ]),
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 0.0, 41.0/840.0, 41.0/840.0]),
    c = np.array([0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 0.5, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0])
)