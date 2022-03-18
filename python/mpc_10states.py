# clear all; close all; clc;

# addpath('/Users/louisjouret/Documents/GitHub/ADCS_LouisJouret/casadi-matlabR2015a-v3.3.0')

import numpy as np
import matplotlib.pyplot as plt

# import casa
from casadi import *


def main():
    dimensions = 3

    ##
    Is1 = 0.05  # inertia of x axis
    Is2 = 0.05  # inertia of y axis
    Is3 = 0.025  # inertia of z axis
    Iw = 3 * 10 ** (-3)  # inertia of reaction wheel
    b = 6 * 10 ** (-6)  # motor friction constant
    Km = 0.03  # motor constant

    ##
    x = MX.sym("x", 10)
    u = MX.sym("u", 3)
    
    ode = [
        0.5 * (-x[1] * x[4] - x[2] * x[5] - x[3] * x[6]),  # q1
        0.5 * (x[0] * x[4] + x[3] * x[5] - x[2] * x[6]),  # q2
        0.5 * (-x[3] * x[4] + x[0] * x[5] + x[1] * x[6]),  # q3
        0.5 * (x[2] * x[4] - x[1] * x[5] + x[0] * x[6]),  # q4
        (1 / Is1)
        * (
            (Is2 - Is3) * x[6] * x[5] - x[5] * x[9] + x[6] * x[8] + b * x[7] - Km * u[0]
        ),  # w1
        (1 / Is2)
        * (
            (Is3 - Is1) * x[4] * x[6] - x[6] * x[7] + x[4] * x[9] + b * x[8] - Km * u[1]
        ),  # w2
        (1 / Is3)
        * (
            (Is1 - Is2) * x[5] * x[4] - x[4] * x[8] + x[5] * x[7] + b * x[9] - Km * u[2]
        ),  # w3
        (1 / Iw) * (Km * u[0] - b * x[7]),  # ww1
        (1 / Iw) * (Km * u[1] - b * x[8]),  # ww2
        (1 / Iw) * (Km * u[2] - b * x[9]), # ww3
    ]  

    # f = Function("f", {x, u}, {ode}, {"x", "u"}, {"ode"})
    f = Function("f", [x, u], ode, ["x", "u"], ["ode"])

    # Time horizon
    T = 1

    # Number of control intervals
    N = 10

    # Time simulated in seconds
    time = 15

    ## casadi variable definition

    # Integrator to discretize the system
    intg_options = struct
    intg_options.tf = T / N
    intg_options.number_of_finite_elements = 4

    # DAE problem structure
    dae = struct
    dae.x = x  # What are states?
    dae.p = u  # What are parameters (=fixed during the integration horizon)?
    dae.ode = f(x, u)  # Expression for the right-hand side

    intg = integrator("intg", "rk", dae, intg_options)
    res = intg("x0", x, "p", u)  # Evaluate with symbols
    
    x_next = res.xf
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
        J = J + Q * ((x[1:, k] - ref).T * (x[:3, k] - ref))
        J = J + 0.001 * Q * (x[5:, k].T * x[5:, k])
        J = J + R * u[:, k].T * u[:, k]

    opti.minimize(J)
    opti.subject_to(-5 <= u <= 5)
    opti.subject_to(x[:, 0] == p)
    opti.solver("ipopt")

    ## log arrays
    X_log = []
    U_log = []
    J_log = []

    ## initial values and reference
    x_new = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    reference = [1, 0, 0, 0]

    ##
    opti.set_value(p, x_new)
    opti.set_value(u_old, np.zeros(dimensions))
    opti.set_value(ref, reference)

    X_log[:, 0] = x_new

    Kwd = 1 * np.eye(dimensions)

    for i in range(time * (N / T)):
        print(i * T)
        sol = opti.solve()
        # u_new   = sol.value(u(:, 1));
        u_new = sol.value(u[:, 0])
        x_new = F(x_new, u_new)
        x_new = full(x_new)
        X_log[:, i + 1] = x_new
        U_log[:, i] = u_new

        opti.set_value(p, x_new)
        opti.set_value(u_old, u_new)
        opti.set_value(ref, reference)

    ## Plotting
    # plot the angular velocity
    t = np.linspace(0, time, 1 + time * (N / T))

    # f = figure('visible','on')
    fig, axs = plt.subplots(3, 1)
    # subplot(3,1,1)
    axs[0].plot(t, X_log[4, :], label="w1")
    # subplot(3,1,2)
    axs[0].plot(t, X_log[5, :], label="w2")
    # subplot(3,1,3)
    axs[0].plot(t, X_log[6, :], label="w3")
    axs[0].set_xlabel("Time [s]")

    # plot the quaternions
    # l = figure('visible','on')
    # subplot(4,1,1)
    fig, axs = plt.subplot(4, 1)
    axs[0].plot(t, X_log[0, :])
    # legend('q0')
    # subplot(4,1,2)
    axs[1].plot(t, X_log[1, :])
    # legend('q1')
    # subplot(4,1,3)
    axs[2].plot(t, X_log[2, :])
    # legend('q2')
    # subplot(4,1,4)
    axs[3].plot(t, X_log[3, :])

    # legend('q3')
    axs[3].set_xlabel("Time [s]")

    # plot the speed of the reaction wheels
    # g = figure('visible','on')
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, X_log[7, :])
    # legend('ww1')
    # subplot(3,1,2)
    axs[1].plot(t, X_log[8, :])
    # legend('ww2')
    # subplot(3,1,3)
    axs[2].plot(t, X_log[9, :])
    # legend('ww3')
    axs[2].set_xlabel("Time [s]")

    # plot the inputs
    # t(:,1) = []
    # h = figure('visible','on')
    fig, axs = plt.subplots(3, 1)
    # subplot(3, 1, 1)
    axs[0].stairs(t, U_log[0, :], label="u1")
    # subplot(3, 1, 2)
    axs[1].stairs(t, U_log[1, :], label="u2")
    # subplot(3, 1, 3)
    axs[2].stairs(t, U_log[2, :], label="u3")
    axs[2].legent()
    axs[2].set_xlabel("time [s]")

    # Plot the kinetic energy
    W = X_log[5:8, :]
    W = W**2
    W = np.sum(W, axis=0)
    W = np.sqrt(W)
    # k = figure('visible','on')

    fig, ax = ax.subplots()
    ax.plot(W)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Kinetic Energy")


if (__name__) == "__main__":
    main()
