% trains RNN to do three bit flip-flop task
% FORCE learning on readout weights W

clear; close all;

% simulation parameters
T = 30000;
p.T = T;
p.do_plot = 0;
p.recordX = 1;

% store the first few weight modifications
p.recordWos = 1;
p.Wo_interval = 50;
p.maxWos = 40;

% net parameters
N=1000;
apply_settings;
nIn = ndim;
nOut = ndim;
g = 1.5;
alpha = 1;
tau = 10;
train_delta = 2;

[input,ft] = gen_stim(T,nOut);
fprintf('Training...\n');

if ndim == 4 && do_bootstrap
    fprintf('Bootstrapping Training...\n');
    load('data/net3.mat','net');
    net.Wfb = [net.Wfb 2.0*(rand(N,1)-0.5)]; % append column
    net.B = [net.B 2.0*(rand(N,1)-0.5)]; % append column
    net.Wo = [net.Wo; zeros(1,N)]; % append row
else
    % training from scratch
    fprintf('Training %d-D Flip-Flop from Scratch...\n',ndim);
    net = init_net(N,nIn,nOut,g,alpha,tau,train_delta,'tanh');
end

[ net, outData ] = force_rls( input, ft, p, net);

% save the weights:
fname = sprintf('data/%s_train.mat',prefix);
save(fname,'net','outData','p','ft','input');

%% PLOTTING TRAIN RESULTS
apply_settings;
fname = sprintf('data/%s_train.mat',prefix);
load(fname);


T=p.T;
nOut = size(outData.Z,1);

figure;
% training results
target_trace = bsxfun(@plus,ft,3*(0:nOut-1)');
output_trace = bsxfun(@plus,outData.Z,3*(0:nOut-1)');
plot(1:T,target_trace,'linewidth',3,'color','green');
plot(1:T,output_trace,'r-');
title('Training Target & Output');

% MSE
figure;
plot(1:2:T,outData.MSE);
title('Training MSE');

% Weights
figure;
plot(1:2:T,outData.Wo_norm);
title('|W_o|');

%% TESTING:

% output
fprintf('Testing...\n');
T = 5000;
[test_input,test_ft] = gen_stim(T,nOut); 
p.T = T;
p.recordX = 1;
p.recordQ = 0;
p.recordR = 0;

outData = test_rnn(test_input, p, net);
MAE_test = mean(mean(((outData.Z - test_ft).^2),2));
fprintf('Test MAE : %f\n', MAE_test);

test_fname = sprintf('data/%s_test.mat',prefix);
save(test_fname,'outData','p','test_ft','test_input');

%% PLOTTING TESTING OUTPUT
figure(2);
target_trace = bsxfun(@plus,test_ft,3*(0:nOut-1)');
output_trace = bsxfun(@plus,outData.Z,3*(0:nOut-1)');
plot(1:T,target_trace,1:T,output_trace);
title('Testing Target & Output');