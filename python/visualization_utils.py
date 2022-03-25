""" Visualization tools for MPC-simulator."""
# Author: Lukas Huber
#    (based on MATLAB script by Louis Journet)  
# Created: 2022-03-25
# Email: lukas.huber@epfl.ch

import matplotlib.pyplot as plt

import numpy as np

def plot_helper_legend_ticks(axs, x_lim):
    for ii, ax in enumerate(axs):
        ax.legend()
        ax.set_xlim(x_lim)

        if ii != len(axs)-1:
            ax.axes.xaxis.set_visible(False)
            
def plot_mpc(t, X_log, U_log):
    plt.ion()
    plt.close('all')
    
    # plot the angular velocity
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, X_log[4, :], label="w1")
    axs[1].plot(t, X_log[5, :], label="w2")
    axs[2].plot(t, X_log[6, :], label="w3")
    
    plot_helper_legend_ticks(axs, [t[0], t[-1]])
    axs[2].set_xlabel("Time [s]")

    # Plot the quaternions
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, X_log[0, :], label="q0")
    axs[1].plot(t, X_log[1, :], label="q1")
    axs[2].plot(t, X_log[2, :], label="q2")
    axs[3].plot(t, X_log[3, :], label="q3")

    plot_helper_legend_ticks(axs, [t[0], t[-1]])
    axs[3].set_xlabel("Time [s]")

    # Plot the speed of the reaction wheels
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, X_log[7, :], label="dw1")
    axs[1].plot(t, X_log[8, :], label="dw2")
    axs[2].plot(t, X_log[9, :], label="dw3")

    plot_helper_legend_ticks(axs, [t[0], t[-1]])
    axs[2].set_xlabel("Time [s]")

    # Plot the inputs
    fig, axs = plt.subplots(3, 1)
    # axs[0].plot(t[:], U_log[0, :], label="u1")
    # axs[1].plot(t[:], U_log[1, :], label="u2")
    # axs[2].plot(t[:], U_log[2, :], label="u3")

    axs[0].stairs(U_log[0, :-1],  t[:], label="u1")
    axs[1].stairs(U_log[1, :-1], t[:], label="u2")
    axs[2].stairs(U_log[2, :-1], t[:] , label="u3")

    for ax in axs:
        ax.set_ylim([-1, 1])
    
    plot_helper_legend_ticks(axs, [t[1], t[-1]])
    axs[2].set_xlabel("Time [s]")

    # Plot the kinetic energy
    W = X_log[5:8, :]
    W = W**2
    W = np.sum(W, axis=0)
    W = np.sqrt(W)

    fig, ax = plt.subplots()
    ax.plot(t, W)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Kinetic Energy")

