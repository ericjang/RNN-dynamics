
nOut=4;

T = 5000;

nTrials=10;
nLevels=20;

[test_input,test_ft] = gen_stim(T,nOut); 
etas = logspace(-4,0,nLevels); % noise levels
p.T = T;
p.recordX = 0;
p.recordQ = 0;
p.recordR = 0;

% default (tanh)

load('data/net4_train.mat','net');
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

% sigmoid
load('data/net4_sigmoid2_train.mat','net'); % overwrite net
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


% bootstrapped tanh
load('data/net4_bs_train.mat','net'); % overwrite net
Wr=net.Wr;
MSE_tanh_bs = zeros(nTrials,nLevels);

for i=1:length(etas)
    eta = etas(i);
    for j=1:nTrials
        fprintf('%d,%d\n',i,j);
        noise = normrnd(0,eta,[N,N]); % added to recurrent matrix
        net.Wr = Wr + noise;
        [outData] = test_rnn(test_input,p,net);
        MSE_tanh_bs(j,i) = mean(mean((outData.Z - test_ft).^2,2));        
    end
end


save('data/MSE_activations_net4.mat','etas','MSE_tanh','MSE_sigmoid','MSE_tanh_bs');

%%
load('data/MSE_activations_net4.mat')
plot(etas,mean(MSE_tanh,1),etas,mean(MSE_sigmoid,1),etas,mean(MSE_tanh_bs,1));
legend({'tanh','sigmoid','tanh bootstrapped'});
xlabel('\eta');
grid on;
ylabel('MSE');
ylim([10^-1.3,10^2]);