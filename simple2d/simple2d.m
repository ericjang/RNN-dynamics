function [q, grad, G] = simple2d(X)
%SIMPLE2D computes objective energy function q, gradient, and Hessian
% for the system shown in 2d_demo

x=X(1);
y=X(2);
xdot = (1-x^2)*y;
ydot = x/2-y;
F = [xdot; ydot]; % F(x) = dx/dt
q = 0.5*(sum(F.^2)); % q = 1/2|F(X)|^2
J = [-2*x*y, 1-x^2; 
      1/2,  -1]; % jacobian
grad = J*F;
G = J'*J; % Gauss-Newton approximation to Hessian
end

