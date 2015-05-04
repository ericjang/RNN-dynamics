% Load fixed J, W, Wfb
clear; close all;
apply_settings;
train_fname = sprintf('data/%s_train.mat',prefix);
test_fname = sprintf('data/%s_test.mat',prefix);
TrainDat = load(train_fname); % net
TestDat = load(test_fname);

net = TrainDat.net; % TrainDat provides the network params
% TestDat provides the ICs for trajectories

N = net.N;

nIC = 20; % number of initial conditions
FPs = zeros(N,nIC);
trajectories = cell(1,nIC);
for i=1:nIC
    fprintf('%d\n',i);
    % pick a random initial trajectory from the test set
    t = randi(TestDat.p.T); 
    x0 = TestDat.outData.X(:,t);
    [FPs(:,i) history] = rnn_findfp(x0,net);
    trajectories{i} = history.x;
end

% compute phase portrait
Wfb = net.Wfb; % feedback
nIC2 = 100;
trajectories2 = cell(1,nIC);
g=net.g;

p.T = 100;
p.do_plot = 0;
p.recordX = 1;
p.recordQ = 1;
p.recordR = 0;

nIn = size(net.B,2);
input = zeros(nIn,p.T);
Qs = cell(1,nIC); % store energy

nOut = size(net.Wo,1);

for i=1:nIC2
    fprintf('%d\n',i);
    z = 2*(rand([nOut,1])-.5); % generate random state in output-space
    % convert it to "internal RNN space"
    % however, Wfb*z only partially accounts for dynamics so we scale it up
    % to fill the cube
    scale = 5*rand();
    net.x0 = scale*Wfb*z; % generate an initial state
    [outData] = test_rnn(input,p,net);
    trajectories2{i} = outData.X;
    Qs{i} = outData.Q;
    %[FPs(:,i) trajectories2{i}] = rnn_findfp(x0,net);
end

% data : rows = observations, columns = variables
data = FPs';

[W,Yfp] = pca(data,'VariableWeights','variance','Centered',true); 
W = diag(std(data))\W;
[~, mu, we] = zscore(data);
we(we==0) = 1;

%save('data/q3.mat','trajectories','trajectories2','Qs','mu','we','W','FPs','Yfp');
fps_fname = sprintf('data/FPs_%s',prefix);
save(fps_fname,'trajectories','trajectories2','Qs','mu','we','W','FPs','Yfp');

%% PCA & Plot Fixed Points

apply_settings;

fps_fname = sprintf('data/FPs_%s.mat',prefix);
load(fps_fname);

figure; hold on;
scatter3(Yfp(:,1),Yfp(:,2),Yfp(:,3),'go'); % plot fixed points

%% Plot Test Trajectory

% compute projection matrix
apply_settings;
train_fname = sprintf('data/%s_train.mat',prefix);
TrainDat = load(train_fname);

% transform new 1000-D points (trajectory) to new space
X = TrainDat.outData.X';
X = bsxfun(@minus,X,mu);
X = bsxfun(@rdivide,X,we);
Y = X*W; % rows = observations, columns = coordinates

plot3(Y(:,1),Y(:,2),Y(:,3),'b'); 

%% plot Q-optimization trajectories with RNN initial states
nIC = size(trajectories,2);
for i=1:nIC
   X = trajectories{i}';
   X = bsxfun(@minus,X,mu);
   X = bsxfun(@rdivide,X,we);
   Y = X*W;
   plot3(Y(:,1),Y(:,2),Y(:,3),'g');
end

%% General Trajectories W/ Random X0
nIC2 = size(trajectories2,2);
for i=1:nIC2
   X = trajectories2{i}';
   X = bsxfun(@minus,X,mu);
   X = bsxfun(@rdivide,X,we);
   Y = X*W;
   xx = Y(:,1);
   yy = Y(:,2);
   zz = Y(:,3);
   
   % compute components
   
   u = xx(2:end)-xx(1:end-1);
   v = yy(2:end)-yy(1:end-1);
   w = zz(2:end)-zz(1:end-1);
   
   quiver3(xx(1:end-1),yy(1:end-1),zz(1:end-1),u,v,w);
   
   %plot3(Y(:,1),Y(:,2),Y(:,3),'r-');
end

%% save images
% set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 4]);
% set(gcf, 'Visible', 'off');
% xlim([-50,50]);
% ylim([-50,50]);
% zlim([-50,50]);
% el=10;
% 
% apply_settings;
% fprefix = sprintf('images/structure_%s',prefix);
% for az=1:360
%     view(az,el);
%     fname = sprintf('%s/%.3d.png',fprefix,az);
%     print(fname,'-dpng','-r100');
%     fprintf('%d\n',az);
% end
% 
