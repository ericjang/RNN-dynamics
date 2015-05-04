% Compute phase trajectories for random initial states
% project each trajectory down to lower-dimensional space
% for each Wo, simulate some trajectories in state-space

% setup
clear; close all;
apply_settings;
fname = sprintf('data/%s_train.mat',prefix);
fprintf([fname '\n']);
TrainDat = load(fname); % network and Wos
fpname = sprintf('data/FPs_%s.mat',prefix);
fprintf([fpname '\n']);
load(fpname); % for the fixed points, W, mu, we

Wos = TrainDat.outData.Wos;
nWos = size(Wos,2);
net = TrainDat.net;
nOut = size(net.Wo,1);

% actual test trajectories, only 1 initial condition
trajectories_pca = cell(nWos,1);

T = 5000;
% testing sequence of choice
[test_input,test_ft] = gen_stim(T,nOut); 
p.T = T;
p.recordX = 1;
p.recordQ = 0;

for i=1:nWos
   fprintf('%d\n',i);
   Wo = Wos{i};
   net.Wo = Wo;
   [outData] = test_rnn(test_input,p,net);
   X = outData.X';
   X = bsxfun(@minus,X,mu);
   X = bsxfun(@rdivide,X,we);
   Y = X*W; % rows = observations, columns = coordinates
   trajectories_pca{i}=Y(:,1:3);
end

% shorter trajectories, sampled randomly from phase space
nIC2 = 30;

p.T = 100;
p.do_plot = 0;
p.recordX = 1;

nIn = size(net.B,2);
input = zeros(nIn,p.T);
trajectories2_pca = cell(nWos,nIC2); % training iter across rows, trajectories across columns for same network
Wfb = net.Wfb;

% fixed set of initial conditions
X0s = zeros(net.N,nIC2);
for j=1:nIC2
    z = 2*(rand([nOut,1])-.5);
    scale = 5*rand();
    X0s(:,j) = scale*Wfb*z;
end

for i=1:nWos
   Wo = Wos{i};
   net.Wo = Wo;
   for j=1:nIC2
       fprintf('i=%d, j=%d\n',i,j);
        net.x0 = X0s(:,j);
        [outData] = test_rnn(input,p,net);
        % compress trajectory to 3 dimensions
        X = outData.X';
        X = bsxfun(@minus,X,mu);
        X = bsxfun(@rdivide,X,we);
        Y = X*W; % rows = observations, columns = coordinates
        trajectories2_pca{i,j}=Y(:,1:3);
   end
end
fname = sprintf('data/%s_train_progression_pca.mat',prefix);
save(fname,'trajectories_pca','trajectories2_pca');


