function [ net ] = init_net( N,nIn,nOut,g,alpha,tau,train_delta, nonlinearity )
%INIT_NET Summary of this function goes here
%   Detailed explanation goes here

net.N = N;
net.Wr = normrnd(0,sqrt(g^2/N),[N,N]); % recurrent
net.Wfb = 2.0*(rand(N,nOut)-0.5); % feedback
net.Wo =  zeros(nOut,N); % readout
net.B = 2.0*(rand(N,nIn)-0.5); % input weights

if strcmp(nonlinearity,'tanh')
    net.act_fun = @(x) tanh(x); 
    net.act_fun_dot = @(x) 1-tanh(x).^2; 
elseif strcmp(nonlinearity,'sigmoid')
    sigmoid = @(x) 1./(1+exp(-x));
    net.act_fun = sigmoid;
    net.act_fun_dot = @(x) sigmoid(x).*(1-sigmoid(x));
elseif strcmp(nonlinearity,'relu')
    % this one blows up, doesn't work
    net.act_fun = @(x) max(0,x);
    net.act_fun_dot = @(x) (x>0);
end

net.alpha = alpha;
net.tau = tau;
net.train_delta = train_delta;
net.g = g;


end
