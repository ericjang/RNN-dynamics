% Similar to the three-bit flip flop, except this trains a 
% 'mode switching' task.

% okay,seems like mode switching doesn't automatically work
% it should suffice to study the phase space of walking / run cycles.

clear; close all;

retrain = 1;

D = load('motions.mat');
motion_train = D.walk';
% truncate the beginning of walk - it's not cyclic
motion_train = motion_train(:,101:480);


T = size(motion_train,2); % size of walk cycle
P_wr = 0;
P_rw = 1;
modes_train = ones(1,T);
%[ modes_train, motion_train ] = gen_stim_motor( T, P_wr, P_rw );

% train
addpath(genpath('../threebit'));
train_delta = 2;
g = 1.5;
N = 1000;
alpha = 1;

% set up parameters
p.P_wr = P_wr;
p.P_rw = P_rw;
p.N = N;
p.g = g;
p.alpha = alpha;
p.T = T;
p.train_delta = train_delta;
p.tau = 10;
p.do_plot = 0;
p.act_fun = @(x) tanh(x);
 
nIn = 1; % 1-D mode input
nOut = size(motion_train,1);
input = modes_train;

if retrain
    load('net_motion.mat','net');
else
    net.Wr = normrnd(0,sqrt(g^2/N),[N,N]); % recurrent
    net.Wfb = 2.0*(rand(N,nOut)-0.5); % feedback
    net.Wo =  zeros(nOut,N); % readout
    net.B = 2.0*(rand(N,nIn)-0.5); % input weights
end

fprintf('Training...\n');
[ net, output, MSE, Wo_norm ] = force_rls( input, motion_train, p, net);
% save the weights:
save('net_motion.mat','net','output','p','MSE','Wo_norm');


%% PLOT RESULTS
subplot(3,1,1);
plot(1:T,motion_train);
hold on;
plot(1:T,output,'linewidth',2,'color','red');
hax = gca();
for i=2:T
   if modes_train(i) ~= modes_train(i-1)
       col = [0 1 0];
       if modes_train(i) == 2
          col = [1 0 0];
       end
       line([i i],get(hax,'YLim'),'Color',col)
   end
end
title('Training Target & Output');

% MSE
subplot(3,1,2);
plot(1:2:T,MSE);
title('MSE');

% Weights
subplot(3,1,3);
plot(1:2:T,Wo_norm);
title('Mean Weight Norm');



%% TESTING
%load('net_motion.mat');

% shorter testing period
%fprintf('Testing...\n');
%[ modes_test, test_ft ] = gen_stim_motor( p.T, p.P_wr, p.P_rw );
%test_out = test_rnn(modes_test, p, net);
%MAE_test = mean(mean(((test_out - test_ft).^2),2));
%fprintf('Test MAE : %f\n', mean(MSE));

%% PLOTTING TESTING OUTPUT
% figure(2);
% hold on;
% plot(1:T,test_ft); 
% plot(1:T,test_out,'r-');
% hax = gca();
% for i=2:T
%    if modes_test(i) ~= modes_test(i-1)
%        col = [0 1 0];
%        if modes_test(i) == 2
%           col = [1 0 0];
%        end
%        line([i i],get(hax,'YLim'),'Color',col)
%    end
% end
% title('Testing Target & Output');
