
%% PCA visualization of testing sequence

% This looks like gibberish

% PCA of trajectory itself
% TrainDat = load('net3.mat'); % net
% data = TrainDat.outData.X';
% [W,Y] = pca(data,'VariableWeights','variance','Centered',true); % data : rows = observations, columns = variables
% plot3(Y(:,1),Y(:,2),Y(:,3));
% [~, mu, we] = zscore(data);
% we(we==0) = 1;

% TestDat = load('net3_test.mat');
% X = TestDat.outData.X';
% X = bsxfun(@minus,X,mu);
% X = bsxfun(@rdivide,X,we);
% Y = X*W; % rows = observations, columns = coordinates
% figure;
% plot3(Y(:,1),Y(:,2),Y(:,3))
% title('Testing Sequence');

% Plot results
%plot(fixed(1),fixed(2),'.r','linewidth',3,'markersize',15);

% jacobian
% x=fixed(1);
% y=fixed(2);
% J = [-2*x*y, 1-x^2; 1/2,  -1]; 
% [v d] = eig(J);
% [x,y]
% fprintf('lambda_1 = %f\n',v(1));
% fprintf('lambda_2 = %f\n',v(2));

% plot data 

% this cell is broken for now

TrainDat = load('net3.mat');
Wos = TrainData.outData.Wos;
net = TrainData.net;
nWos = size(Wos,2);

% plot test MAE as a function of training iter

testT = size(Zs{1},2);
trainT = length(train_MSE);
Wo_interval = 1000;
train_iters = (1:nWos) * Wo_interval;
nOut = size(net.Wo,1);

test_MSE = zeros(1,nWos);
for i=1:nWos
    test_MSE(i) = mean(mean(((Zs{i} - FTs{i}).^2),2));
end

%% plot Train MSE and Test MSE over iterations
figure;
plot(1:trainT,train_MSE,train_iters,test_MSE);
legend({'Train MSE','Test MSE'});
xlabel('Training Iterations');
ylabel('MSE');


%% plot Z vs FT
figure;
for i=1:nWos
   subplot(1,nWos,i);
   target_trace = bsxfun(@plus,FTs{i},3*(0:nOut-1)');
   output_trace = bsxfun(@plus,Zs{i},3*(0:nOut-1)');
   plot(1:testT,target_trace,1:testT,output_trace);
end