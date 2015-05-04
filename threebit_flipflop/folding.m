
clear; close all;

N=100;
nIn = 3;
nOut = 3;
g = 1.5;
alpha = 1;
tau = 10;
train_delta = 2;


% initial weights, conditions identical
tanh_net = init_net(N,nIn,nOut,g,alpha,tau,train_delta,'tanh');
x0 = linspace(-1,1,N)';
tanh_net.x0 = x0(randperm(N));

sigmoid_net = tanh_net;
sigmoid = @(x) 1./(1+exp(-x));
sigmoid_net.act_fun = sigmoid;
sigmoid_net.act_fun_dot = @(x) sigmoid(x).*(1-sigmoid(x));

% simulation parameters
T = 50;
p.T = T;
p.recordX = 1;
p.recordR = 1;
p.do_plot = 0;
p.recordWos = 0;
p.recordQ = 0;

tt=sin((1:p.T)/10);
input = repmat(tt,nIn,1);

%input=zeros(nIn,T);

tanh_out = test_rnn(input, p, tanh_net);
sigmoid_out = test_rnn(input, p, sigmoid_net);

ttrace = 1:.5:p.T+.5;
tanh_XR = zeros(N,length(ttrace));
tanh_XR(:,1:2:p.T*2-1) = tanh_out.X; % odds
tanh_XR(:,2:2:p.T*2) = tanh_out.R; % evens

sigmoid_XR = zeros(N,length(ttrace));
sigmoid_XR(:,1:2:p.T*2-1) = sigmoid_out.X; % odds
sigmoid_XR(:,2:2:p.T*2) = sigmoid_out.R; % evens

figure; hold on;
ax1 = plot(ttrace,tanh_XR','g.',ttrace,sigmoid_XR','r.');
legend({'tanh','sigmoid'});
title('Mixing with Periodic Input');
%plot(ttrace, sigmoid_XR);




