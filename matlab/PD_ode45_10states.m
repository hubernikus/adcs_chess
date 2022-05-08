
clear all;
close all;
%%
h = figure('visible','off');

fps = 50;
time = 30;
%% system parameters

Is1 = 0.065;
Is2 = 0.065;
Is3 = 0.037;
Iw  = 3*10^(-3);
b   = 6*10^(-5);
Km  = 0.03;

x_new = [0;1;0;0;1;-1;1;0;0;0];
u_new = [0;0;0];

h = 1/fps; %sampling period
I = diag([Is1,Is2,Is3]);

ref = [1;0;0;0];

X=[x_new];
U=[u_new];

Kp = 2;
Kd = 1;

f = @(t,x,u)[0.5*(-x(2)*x(5) - x(3)*x(6) - x(4)*x(7));
            0.5*(x(1)*x(5) + x(4)*x(6) - x(3)*x(7));
            0.5*(-x(4)*x(5) + x(1)*x(6) + x(2)*x(7));
            0.5*(x(3)*x(5) - x(2)*x(6) + x(1)*x(7));
            (1/Is1)*((Is2-Is3)*x(7)*x(6) - Iw*x(6)*x(10) + Iw*x(7)*x(9) + b*x(8) - Km*u(1));
            (1/Is2)*((Is3-Is1)*x(5)*x(7) - Iw*x(7)*x(8) + Iw*x(5)*x(10) + b*x(9) - Km*u(2));
            (1/Is3)*((Is1-Is2)*x(6)*x(5) - Iw*x(5)*x(9) + Iw*x(6)*x(8) + b*x(10) - Km*u(3));
            (1/Iw)*(Km*u(1) - b*x(8));
            (1/Iw)*(Km*u(2) - b*x(9));
            (1/Iw)*(Km*u(3) - b*x(10))];

for i=1:time*fps
    i
    %% computing the controller
    q_err = quatmultiply(quatconj(ref.'), x_new(1:4).' );
    q_err = q_err(2:4).';
    w_err = x_new(5:7);
        
    u_new =  Kp*q_err + Kd*w_err;

    %% solve the differential equation
    [t,x] = ode45(@(t,x) f(t,x,u_new) , [0 h] , x_new);
    x_new = x(end,:).';
    
    %% check if the norme of quaternions is still 1
    norme = quatmod([x_new(1), x_new(2), x_new(3), x_new(4)]);
    if norme>2
        disp('error: norme too high')
        break;
    end
    X = [X x_new];
    U = [U u_new];
    %x_new(1:4) = (1/norme) .* x_new(1:4);
end

%% plotting

t = linspace(0,time,1+time*fps);
f = figure('visible','on');
subplot(3,1,1);
plot(t,X(5,:));
legend('wsatx');
subplot(3,1,2);
plot(t,X(6,:));
legend('wsaty');
subplot(3,1,3);
plot(t,X(7,:));
legend('wsatz');
xlabel('time [s]');

g = figure('visible','on');
subplot(3,1,1);
plot(t,X(8,:));
legend('ww1');
subplot(3,1,2);
plot(t,X(9,:));
legend('ww2');
subplot(3,1,3);
plot(t,X(10,:));
legend('ww3');
xlabel('time [s]');


l = figure('visible','on');
subplot(4,1,1);
plot(t,X(1,:));
legend('q0');
subplot(4,1,2);
plot(t,X(2,:));
legend('q1');
subplot(4,1,3);
plot(t,X(3,:));
legend('q2');
subplot(4,1,4);
plot(t,X(4,:));
legend('q3');
xlabel('time [s]');

h = figure('visible','on');
subplot(3,1,1);
stairs(t,U(1,:));
legend('u1');
subplot(3,1,2);
stairs(t,U(2,:));
legend('u2');
subplot(3,1,3);
stairs(t,U(3,:));
legend('u3');
xlabel('time [s]');


W = X(5:7,:);
W = W.^2;
W = sum(W,1);
W = sqrt(W);
k = figure('visible','on');
plot(W);
xlabel('time [s]');
ylabel('Kinetic Energy');