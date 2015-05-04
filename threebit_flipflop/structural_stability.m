% test which of the tanh vs. sigmoid nonlinearities make the network more
% robust to noise.

% simulation params

nOut=3;

T = 5000;

nTrials=10;
nLevels=20;

[test_input,test_ft] = gen_stim(T,nOut); 
etas = logspace(-4,0,nLevels); % noise levels
p.T = T;
p.recordX = 0;
p.recordQ = 0;
p.recordR = 0;

% Tanh

load('data/net3_tanh_train.mat','net');
N=net.N;
Wr=net.Wr;
MSE_tanh = zeros(nTrials,nLevels);

for i=1:length(etas)
    eta = etas(i);
    for j=1:nTrials
        fprintf('%d,%d\n',i,j);
        noise = normrnd(0,eta,[N,N]); % added to recurrent matrix
        net.Wr = Wr + noise;
        [outData] = test_rnn(test_input,p,net);
        MSE_tanh(j,i) = mean(mean((outData.Z - test_ft).^2,2));        
    end
end

% Sigmoid
load('data/net3_sigmoid_train.mat','net'); % overwrite net
Wr=net.Wr;
MSE_sigmoid = zeros(nTrials,nLevels);

for i=1:length(etas)
    eta = etas(i);
    for j=1:nTrials
        fprintf('%d,%d\n',i,j);
        noise = normrnd(0,eta,[N,N]); % added to recurrent matrix
        net.Wr = Wr + noise;
        [outData] = test_rnn(test_input,p,net);
        MSE_sigmoid(j,i) = mean(mean((outData.Z - test_ft).^2,2));        
    end
end

save('MSE_activations.mat','etas','MSE_tanh','MSE_sigmoid');

plot(etas,mean(MSE_tanh,1),etas,mean(MSE_sigmoid,1));
legend({'tanh','sigmoid'});
xlabel('\eta');
ylabel('MSE');




