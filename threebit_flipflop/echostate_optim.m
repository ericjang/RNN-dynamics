function [ q, grad, G ] = echostate_optim( x, u, net )
%ECHOSTATE_OPTIM Summary of this function goes here
%   Detailed explanation goes here
%    r = tanh(x);

    N=net.N;
    Wr=net.Wr;
    Wfb=net.Wfb;
    Wo=net.Wo;
    B=net.B;
    act_fun=net.act_fun;
    act_fun_dot=net.act_fun_dot;

    r = act_fun(x);
    W_c = Wr+Wfb*Wo; % combined weight matrix
    F = -x+W_c*r+B*u; % F(x)
    q = 0.5*(sum(F.^2)); % q = 1/2|F(X)|^2
    J = W_c.*(ones(N,1)*act_fun_dot(x)')-eye(N); % Jacobian
    grad = J'*F; % Gradient
    G = J'*J; % Gauss-Newton approximation to Hessian
end
