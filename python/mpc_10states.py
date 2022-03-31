""" Library creating and MPC-environment using Casadi."""
# Author: Lukas Huber
#    (based on MATLAB script by Louis Journet)  
# Created: 2022-03-15
# Email: lukas.huber@epfl.ch

import numpy as np

from casadi import *

from visualization_utils import plot_mpc


def main(T=1, N=10, time=15):
    """
    Inputs
    ------
    T: Time horizon
    N: Number of control intervals
    time: Time simulated in seconds

    Returns
    -------
    [TODO]
    """
    dimensions = 3

    # Physical properties of the system
    Is1 = 0.05  # inertia of x axis
    Is2 = 0.05  # inertia of y axis
    Is3 = 0.025  # inertia of z axis
    Iw = 3 * 10 ** (-3)  # inertia of reaction wheel
    b = 6 * 10 ** (-6)  # motor friction constant
    Km = 0.03  # motor constant

    size_x = 10
    size_u = 3
    size_dx = size_x
    
    ##
    x = MX.sym("x", size_x)
    u = MX.sym("u", size_u)

    ode = MX(size_dx, 1)
    # ode = [None]*10
    # Quaternion
    ode[0] = 0.5 * (-x[1] * x[4] - x[2] * x[5] - x[3] * x[6])
    ode[1] = 0.5 * (x[0] * x[4] + x[3] * x[5] - x[2] * x[6])
    ode[2] = 0.5 * (-x[3] * x[4] + x[0] * x[5] + x[1] * x[6])
    ode[3] = 0.5 * (x[2] * x[4] - x[1] * x[5] + x[0] * x[6])

    # Angular Velocity
    ode[4] = ((1 / Is1) 
        * (
            (Is2 - Is3) * x[6] * x[5] - x[5] * x[9] + x[6] * x[8] + b * x[7] - Km * u[0]
        ))
    ode[5] = ((1 / Is2)
             * ((Is3 - Is1) * x[4] * x[6] - x[6] * x[7] + x[4] * x[9] + b * x[8] - Km * u[1])
             )
    ode[6] = ((1 / Is3)
             * ((Is1 - Is2) * x[5] * x[4] - x[4] * x[8] + x[5] * x[7] + b * x[9] - Km * u[2])
             )

    # Angular Acceleration
    ode[7] = ((1 / Iw) * (Km * u[0] - b * x[7]))
    ode[8] = ((1 / Iw) * (Km * u[1] - b * x[8]))
    ode[9] = ((1 / Iw) * (Km * u[2] - b * x[9]))

    f = Function("f", [x, u], [ode], ["x", "u"], ["ode"])
    
    ## casadi variable definition

    # Integrator to discretize the system
    intg_options = {"tf": T / N, "number_of_finite_elements": 4}

    # DAE problem structure
    dae = {
        # What are states?
        "x": x,
        # What are parameters (=fixed during the integration horizon)?
        "p": u,
        # Expression for the right-hand side
        "ode": f(x, u),
        }

    intg = integrator("intg", "rk", dae, intg_options)
    # res = intg("x0", x, "p", u)  # Evaluate with symbols
    res = intg(x0=x, p=u)  # Evaluate with symbols
    
    x_next = res['xf']
    F = Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

    ## optimization loop
    opti = casadi.Opti()
    x = opti.variable(10, N + 1)  # Decision variables for state trajectory
    u = opti.variable(3, N)
    p = opti.parameter(10, 1)  # Parameter (not optimized over)
    ref = opti.parameter(4, 1)
    u_old = opti.parameter(3, 1)

    Q = 10
    R = 1

    J = 0

    for k in range(N):
        opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

        # compute cost function
        J = J + Q * ((x[:4, k] - ref).T @ (x[:4, k] - ref))
        J = J + 0.001 * Q * (x[5:, k].T @ x[5:, k])
        J = J + R * u[:, k].T @ u[:, k]

    opti.minimize(J)
    # opti.subject_to(-5 <= u)
    # opti.subject_to(-5 <= u <= 5)
    # opti.subject_to(opti.bounded(-5, vec(u), 5))
    opti.subject_to(opti.bounded(-5, u, 5))
    
    opti.subject_to(x[:, 0] == p)
    opti.solver("ipopt")

    ## log arrays
    n_steps = int(time * (N / T))
    X_log = []
    U_log = []
    J_log = []

    ## initial values and reference
    x_new = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    reference = [1, 0, 0, 0]

    ##
    opti.set_value(p, x_new)
    opti.set_value(u_old, np.zeros(dimensions))
    opti.set_value(ref, reference)

    X_log.append(x_new)
    U_log.append(np.zeros(size_u))
    
    Kwd = 1 * np.eye(dimensions)

    for i in range(n_steps):
        print(i * T)
        sol = opti.solve()
        # u_new   = sol.value(u(:, 1));
        u_new = sol.value(u[:, 0])
        x_new = F(x_new, u_new)
        # x_new = full(x_new)
        X_log.append(np.squeeze(x_new))
        U_log.append(np.squeeze(u_new))

        opti.set_value(p, x_new)
        opti.set_value(u_old, u_new)
        opti.set_value(ref, reference)

    X_log = np.array(X_log).T
    U_log = np.array(U_log).T

    t = np.linspace(0, time, n_steps+1)
    
    return t, X_log, U_log


if (__name__) == "__main__":
    t, X_log, U_log = main(T=1, N=10, time=15)
    plot_mpc(t, X_log, U_log)
