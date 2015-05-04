function [ J ] = esn_jacobian( x, net )
%RNN_JACOBIAN Computes jacobian of phase portrait about a point x
% useful for linear stability analysis of FPs.

    Wr = net.Wr;
    Wfb = net.Wfb;
    Wo = net.Wo;
    N = net.N;
    act_fn_dot = net.act_fun;
    act_fn = net.act_fun_dot;

    r = act_fun(x);
    W_c = Wr+Wfb*Wo; % combined weight matrix
    J = W_c.*(ones(N,1)*act_fun_dot(x)')-eye(N); % Jacobian
end
