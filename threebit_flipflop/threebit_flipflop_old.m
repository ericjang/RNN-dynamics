% MODIFIED FROM ORIGINAL TO SEE IF THERE IS SOMETHING INHERENTLY WRONG WITH
% MY CODE

disp('Clearing workspace.');
clear; close all;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 0.1;
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 1440;
nOut=3;



nIn=nOut;

dt = 0.1;
learn_every = 2;

scale = 1.0/sqrt(p*N);
M = sprandn(N,N,p)*g*scale; % = J
M = full(M);


nRec2Out = N;
wo = zeros(nOut,nRec2Out); % readout weights (to learn)
dw = zeros(nOut,nRec2Out);
wf = 2.0*(rand(N,nOut)-0.5); % feedback weights

disp(['   N: ', num2str(N)]);
disp(['   g: ', num2str(g)]);
disp(['   p: ', num2str(p)]);
disp(['   nRec2Out: ', num2str(nRec2Out)]);
disp(['   alpha: ', num2str(alpha,3)]);
disp(['   nsecs: ', num2str(nsecs)]);
disp(['   learn_every: ', num2str(learn_every)]);


simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

B = 2.0*(rand(N,nIn)-0.5); 

[U,ft] = gen_stim(length(simtime),nOut); 

%plot the stim
%plot(simtime,bsxfun(@plus,ft,3*[0:nOut-1]'));


wo_len = zeros(1,simtime_len); % square root of sum of squared weights
zt = zeros(nOut,simtime_len); % training output
zpt = zeros(nOut,simtime_len); % testing output
x0 = 0.5*randn(N,1); % initial activations
z0 = 0.5*randn(nOut,1); % inital readout

x = x0; 
r = tanh(x);
z = z0; 

figure;
ti = 0;
P = (1.0/alpha)*eye(nRec2Out);

subplot 211;
% target function and predicted output
%plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
plot(simtime,bsxfun(@plus,ft,3*[0:nOut-1]'),'color','green');

do_plot = 1;

for t = simtime
    ti = ti+1;	
    
    
    if do_plot && mod(ti, nsecs/2) == 0
        disp(['time: ' num2str(t,3) '.']);
        subplot 211;
        hold on;
        %plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
        plot(simtime,bsxfun(@plus,zt,3*[0:nOut-1]'),'color','red');
        title('training', 'fontsize', fontsize, 'fontweight', fontweight);
        legend('f', 'z');	
        xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
        ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
        hold off;

        % weight
        subplot 212;
        plot(simtime, wo_len, 'linewidth', linewidth);
        xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
        ylabel('|w|', 'fontsize', fontsize, 'fontweight', fontweight);
        legend('|w|');
        pause(0.5);	
    end
    
    % external input
    u = U(:,ti);
    
    xdot = -x+M*r+wf*z+B*u;
    x = x + xdot*dt;
    
    % sim, so x(t) and r(t) are created.
    %x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo*r;
    
    if mod(ti, learn_every) == 0
	% update inverse correlation matrix
	k = P*r;
	rPr = r'*k;
	c = 1.0/(1.0 + rPr);
	P = P - k*(k'*c);
    
	% update the error for the linear readout
	e = z-ft(:,ti);
	
	% update the output weights
	%dw = -e*k*c;	
	dw = -e*k'*c;
    wo = wo + dw;
    end
    
    % Store the output of the system.
    zt(:,ti) = z;
    wo_len(ti) = sqrt(sum(sum(wo.^2)));	
end
error_avg = mean(sum(abs(zt-ft),2)/simtime_len);
disp(['Training MAE: ' num2str(error_avg,3)]);    
disp(['Now testing... please wait.']);    


% Now test. 
[U_test,ft_test] = gen_stim(length(simtime),nOut); 

ti = 0;
for t = simtime				% don't want to subtract time in indices
    ti = ti+1;    
    
    u = U_test(:,ti);
    
    xdot = -x+M*r+wf*z+B*u;
    x = x + xdot*dt;
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo*r;
    
    zpt(:,ti) = z;
end
error_avg = mean(sum(abs(zpt-ft_test),2)/simtime_len);
disp(['Testing MAE: ' num2str(error_avg,3)]);

figure;
subplot 211;
plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
hold on;
plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
title('training', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
hold on;
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');


subplot 212;
hold on;
plot(simtime2, ft_test, 'linewidth', linewidth, 'color', 'green'); 
axis tight;
plot(simtime2, zpt, 'linewidth', linewidth, 'color', 'red');
axis tight;
title('testing', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');
	

