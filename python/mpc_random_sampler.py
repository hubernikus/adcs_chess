""" Library creating and MPC-environment using Casadi."""
# Author: Lukas Huber
#    (based on MATLAB script by Louis Journet)  
# Created: 2022-03-15
# Email: lukas.huber@epfl.ch
import datetime
import csv

import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation

# For maximum MATLAB compatibility, we keep this for the moment open. 
from casadi import *
# import casadi as cas

from visualization_utils import plot_mpc


class RandomizerMPC:
    quaternion_range = [0, 1]
    # ang_vel_max = 5
    # ang_acc_max = 5
    # OR -> angle = [-pi, pi] + vector = [1, 1, 1]
    
    # Angular Velocity & Angular Acceleration
    angular_velocity_range = [-5, 5]
    angular_acceleration_range = [-5, 5]
    
    def __init__(self, T=1, N=10, time=15):
        self.T = T
        self.N = N
        self.time = time
        
        self.dimensions = 3
        
        self.size_x = 10
        self.size_u = 3
        self.size_dx = self.size_x

        self.setup()

    def setup(self):
        # Physical properties of the system
        Is1 = 0.05  # inertia of x axis
        Is2 = 0.05  # inertia of y axis
        Is3 = 0.025  # inertia of z axis
        Iw = 3 * 10 ** (-3)  # inertia of reaction wheel
        b = 6 * 10 ** (-6)  # motor friction constant
        Km = 0.03  # motor constant

        ## 
        x = MX.sym("x", self.size_x)
        u = MX.sym("u", self.size_u)

        ode = MX(self.size_dx, 1)
        
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
        intg_options = {"tf": self.T / self.N, "number_of_finite_elements": 4}

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
        self.F = Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

        ## optimization loop
        opti = casadi.Opti()

        # Decision variables for state trajectory
        self.x = opti.variable(10, self.N + 1)  
        self.u = opti.variable(3, self.N)
        
        # Parameter (not optimized over)
        self.p = opti.parameter(10, 1)  
        self.ref = opti.parameter(4, 1)
        self.u_old = opti.parameter(3, 1)

        Q = 10
        R = 1

        J = 0

        for k in range(self.N):
            opti.subject_to(self.x[:, k + 1] == self.F(self.x[:, k], self.u[:, k]))

            # compute cost function
            J = J + Q * ((self.x[:4, k] - self.ref).T @ (self.x[:4, k] - self.ref))
            J = J + 0.001 * Q * (self.x[5:, k].T @ self.x[5:, k])
            J = J + R * self.u[:, k].T @ self.u[:, k]

        opti.minimize(J)
        opti.subject_to(opti.bounded(-5, self.u, 5))

        opti.subject_to(self.x[:, 0] == self.p)
        opti.solver("ipopt")

        # Store optimizer as attribute for future computation
        self.opti = opti
        
    def optimize_and_integrate(self, start_position, start_reference):
        x_new = start_position
        reference = start_reference
        
        self.opti.set_value(self.p, x_new)
        self.opti.set_value(self.u_old, np.zeros(self.dimensions))
        self.opti.set_value(self.ref, reference)

        ## Log arrays
        n_steps = int(self.time * (self.N / self.T))
        print(f"Number of steps: {n_steps}.")
        
        X_log = []
        U_log = []
        J_log = []

        X_log.append(x_new)
        U_log.append(np.zeros(self.size_u))

        Kwd = 1 * np.eye(self.dimensions)

        for i in range(n_steps):
            print(f"Time: {i * self.T}")
            sol = self.opti.solve()
            
            # u_new   = sol.value(u(:, 1));
            u_new = sol.value(self.u[:, 0])
            x_new = self.F(x_new, u_new)
            # x_new = full(x_new)
            
            X_log.append(np.squeeze(x_new))
            U_log.append(np.squeeze(u_new))

            self.opti.set_value(self.p, x_new)
            self.opti.set_value(self.u_old, u_new)
            self.opti.set_value(self.ref, reference)

        # Set and store values for plotting and saving to file
        self.t = np.linspace(0, self.time, n_steps+1)
        
        self.x_log = np.array(X_log).T
        self.u_log = np.array(U_log).T

    def plot(self):
        plot_mpc(t=self.t, X_log=self.x_log, U_log=self.u_log)

    def do_random_runs(self, n_points=3, save_to_file=True, filename=None):
        """ Method which creates random samples in ranges and stores it to csv-fle."""

        if filename is None:
            now = datetime.datetime.now()
            filename = f"mpc_learning_{now:%Y-%m-%d_%H-%M-%S}" + ".csv"

        writer = csv.writer(open(filename, 'w'))
        writer.writerow(["q0", "q1", "q2", "q3", "w0", "w1", "w2", "a0", "a1", "a2", "u0", "u1", "u2", "u3"])
            
        # Create header
        for ii in range(n_points):
            quat_random = Rotation.random().as_quat()
            
            angular_vel = (
                np.random.rand(3)*(self.angular_velocity_range[1]
                             - self.angular_velocity_range[0])
                + self.angular_velocity_range[0]
            )

            angular_acc = (
                np.random.rand(3)*(
                    self.angular_acceleration_range[1]
                    - self.angular_acceleration_range[0]
                ) + self.angular_acceleration_range[0]
            )

            x_new = np.hstack((quat_random, angular_vel, angular_acc))

            # TODO: should desired reference be closed point(?)
            start_reference = [1, 0, 0, 0]
            self.opti.set_value(self.p, x_new)
            self.opti.set_value(self.u_old, np.zeros(self.dimensions))
            self.opti.set_value(self.ref, start_reference)

            # Solve system
            sol = self.opti.solve()
            u_new = sol.value(self.u[:, 0])

            # Write to file
            writer.writerow(np.hstack((x_new, u_new)))
            

if (__name__) == "__main__":
    my_sampler = RandomizerMPC()
    my_sampler.do_random_runs(save_to_file=True)
    print("\n\n\n")
    print("Writing finished. \n\n\n")
    
    # my_sampler.optimize_and_integrate(
        # start_position=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        # start_reference=[1, 0, 0, 0],
    # )
    # my_sampler.plot()

