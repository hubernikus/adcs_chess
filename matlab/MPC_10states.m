clear all; close all; clc;

addpath('/home/lukas/Code/adcs_chess/matlab/casadi-linux-octave-5.2.0-v3.5.5')
import casadi.*

%%
Is1 = 0.05;         % inertia of x axis
Is2 = 0.05;         % inertia of y axis 
Is3 = 0.025;        % inertia of z axis
Iw  = 3*10^(-3);    % inertia of reaction wheel
b   = 6*10^(-6);    % motor friction constant
Km  = 0.03;         % motor constant


%%
x = MX.sym('x',10);
u = MX.sym('u',3);

ode= [      0.5*(-x(2)*x(5) - x(3)*x(6) - x(4)*x(7));   % q1
            0.5*(x(1)*x(5) + x(4)*x(6) - x(3)*x(7));    % q2
            0.5*(-x(4)*x(5) + x(1)*x(6) + x(2)*x(7));   % q3
            0.5*(x(3)*x(5) - x(2)*x(6) + x(1)*x(7));    % q4
            (1/Is1)*((Is2-Is3)*x(7)*x(6) - x(6)*x(10) + x(7)*x(9) + b*x(8) - Km*u(1));  % w1
            (1/Is2)*((Is3-Is1)*x(5)*x(7) - x(7)*x(8) + x(5)*x(10) + b*x(9)  - Km*u(2)); % w2
            (1/Is3)*((Is1-Is2)*x(6)*x(5) - x(5)*x(9) + x(6)*x(8) +b*x(10) - Km*u(3));   % w3
            (1/Iw)*(Km*u(1) - b*x(8));  % ww1
            (1/Iw)*(Km*u(2) - b*x(9));  % ww2
            (1/Iw)*(Km*u(3) - b*x(10))];% ww3


f = Function('f',{x,u},{ode},{'x','u'},{'ode'});


T   = 1;    % Time horizon
N   = 10;   % Number of control intervals
time= 15;   % time simulated in seconds

%% casadi variable definition

% Integrator to discretize the system
intg_options = struct;
intg_options.tf = T/N;
intg_options.number_of_finite_elements = 4;

% DAE problem structure
dae    = struct;
dae.x  = x;         % What are states?
dae.p  = u;         % What are parameters (=fixed during the integration horizon)?
dae.ode= f(x,u);  % Expression for the right-hand side

intg   = integrator('intg','rk',dae,intg_options);
res    = intg('x0',x,'p',u); % Evaluate with symbols
x_next = res.xf;
F      = Function('F',{x,u},{x_next},{'x','u'},{'x_next'});

%% optimization loop

opti = casadi.Opti();
x    = opti.variable(10,N+1); % Decision variables for state trajectory
u    = opti.variable(3,N);
p    = opti.parameter(10,1);  % Parameter (not optimized over)
ref  = opti.parameter(4,1);
u_old= opti.parameter(3,1);

Q = 10; 
R = 1;

J = 0;

for k=1:N
    opti.subject_to(x(:,k+1)==F(x(:,k),u(:,k)));
    
    % compute cost function
    J = J + Q*((x(1:4,k) - ref).'*(x(1:4,k) - ref));
    J = J + 0.001*Q*(x(5:10,k).'*x(5:10,k));
    J = J + R*u(:,k).'*u(:,k);
end



opti.minimize(J);
opti.subject_to(-5 <= u <= 5);
opti.subject_to(x(:,1) == p);
opti.solver('ipopt');


%% log arrays
X_log = [];
U_log = [];
J_log = [];

%% initial values and reference
x_new= [0;1;0;0;0;0;0;0;0;0];
reference  = [1;0;0;0];

%%
opti.set_value(p,x_new);
opti.set_value(u_old,[0;0;0]);
opti.set_value(ref,reference);

X_log(:,1) = x_new;

Kwd = 1*eye(3);

for i=1:time*(N/T)
    i*T
    sol     = opti.solve();
    u_new   = sol.value(u(:,1));
    x_new   =  F(x_new,u_new);
    x_new   = full(x_new);
    X_log(  :,i+1) = x_new;
    U_log(  :,i) = u_new;

    opti.set_value(p,x_new); 
    opti.set_value(u_old,u_new);
    opti.set_value(ref,reference);
    
end


%% plotting
% plot the angular velocity
t = linspace(0,time,1+time*(N/T));
f = figure('visible','on');
subplot(3,1,1);
plot(t,X_log(5,:));
legend('w1');
subplot(3,1,2);
plot(t,X_log(6,:));
legend('w2');
subplot(3,1,3);
plot(t,X_log(7,:));
legend('w3');
xlabel('time [s]');

% plot the quaternions
l = figure('visible','on');
subplot(4,1,1);
plot(t,X_log(1,:));
legend('q0');
subplot(4,1,2);
plot(t,X_log(2,:));
legend('q1');
subplot(4,1,3);
plot(t,X_log(3,:));
legend('q2');
subplot(4,1,4);
plot(t,X_log(4,:));
legend('q3');
xlabel('time [s]');

% plot the speed of the reaction wheels
g = figure('visible','on');
subplot(3,1,1);
plot(t,X_log(8,:));
legend('ww1');
subplot(3,1,2);
plot(t,X_log(9,:));
legend('ww2');
subplot(3,1,3);
plot(t,X_log(10,:));
legend('ww3');
xlabel('time [s]');

% plot the inputs
t(:,1) = [];
h = figure('visible','on');
subplot(3,1,1);
stairs(t,U_log(1,:));
legend('u1');
subplot(3,1,2);
stairs(t,U_log(2,:));
legend('u2');
subplot(3,1,3);
stairs(t,U_log(3,:));
legend('u3');
xlabel('time [s]');

% plot the kinetic energy
W = X_log(5:7,:);
W = W.^2;
W = sum(W,1);
W = sqrt(W);
k = figure('visible','on');
plot(W);
xlabel('time [s]');
ylabel('Kinetic Energy');
