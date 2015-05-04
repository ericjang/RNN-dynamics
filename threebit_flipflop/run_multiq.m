% analyzes stability of fixed points
% as function of training iteration

clear; close all;
TrainDat = load('data/net3.mat'); % network and Wos
TestDat = load('data/net3_test.mat'); % test trajectories for ICs

Wos = TrainDat.outData.Wos;
nWos = size(Wos,2);
net = TrainDat.net;
nOut = size(net.Wo,1);

% for each Wo, perform q-optimization to locate fixed points.
% then do eigenvalue analysis.

N = net.N;
nIC = 10; % number of initial conditions

FPs = zeros(nWos,nIC,N);
for i=1:nWos
   for j=1:nIC
       fprintf('%d, %d\n',i,j);
       t = randi(TestDat.p.T); 
       x0 = TestDat.outData.X(:,t);
       [x_star] = rnn_findfp(x0,net);
       FPs(i,j,:) = x_star;
   end
end
save('esn_fps.mat','FPs');
