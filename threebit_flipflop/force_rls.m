function [ net, outData ] = force_rls( input, ft, params, net)
%FORCE_RLS FORCE learning using RLS to update readout weights for an echo-state RNN architecture.
%   INPUTS:
%       input (nIn x nsteps)
%       ft (nOut x nsteps)
%   returns: 
%       Wo - readout weights
%       zt - network output
%       MSE - mean square error of outputs

T = params.T; % number of timesteps

do_plot = params.do_plot; % enable interactive plotting

progress_interval = ceil(T/100);


% Initialize Network 
nIn = size(input,1);

nOut = size(ft,1);

N = net.N;
alpha = net.alpha;
tau = net.tau;
train_delta = net.train_delta; % learning timestep
act_fun = net.act_fun; % activation nonlinearity
Wfb = net.Wfb;
Wo = net.Wo;
Wr = net.Wr;
B = net.B;

assert(nIn==size(B,2));

% initial state
x = 0.5*randn(N,1);
P = (1.0/alpha)*eye(N);

% Train

% plotting params

linewidth = 2;
fontsize = 14;
fontweight = 'bold';

% set up data to be recorded
outData.Z=zeros(nOut,T);
outData.MSE=zeros(1,T/train_delta);
outData.Wo_norm=zeros(nOut,T/train_delta); % norm of readout weights
if params.recordX
    outData.X = zeros(N,T);
end

if params.recordWos
   nWos = min(T/params.Wo_interval,params.maxWos);
   outData.Wos = cell(1,nWos);
   i=1;
end


for ti=1:T
  % update x,r,z
  r = act_fun(x);
  z = Wo*r;
  u = input(:,ti);
  xdot = -x+Wr*r+Wfb*z+B*u;
  x = x + xdot/tau; % integrate with time constant
  
  if mod(ti,train_delta) == 0 % do learning
    % compute P(t) from P(t-deltaT)  
    k = P*r;
	rPr = r'*k;
	c = 1.0/(1.0 + rPr);
	P = P - k*(k'*c);
    % update the error for the linear readout
    e = z-ft(:,ti);
    % update output weights using new P(t)
    dW = -e*k'*c; % the c term is not mentioned in the paper...
    Wo = Wo + dW;
    % record mean weights
    outData.Wo_norm(:,ti/train_delta) = sqrt(sum(Wo.^2,2));
    outData.MSE(ti/train_delta) = mean(e.^2);
  end
  % record output
  outData.Z(:,ti)=z;
  
  if params.recordX
      outData.X(:,ti) = x;
  end
  
  % record Wo matrix every once in awhile
  if params.recordWos && mod(ti, params.Wo_interval) == 0 && i <= params.maxWos 
     outData.Wos{i} = Wo; 
     i = i+1;
  end
  
  if do_plot
    subplot 311;
    hold on;
    plot(1:T,bsxfun(@plus,output,3*(0:nOut-1)'),'color','red');
    %title('training', 'fontsize', fontsize, 'fontweight', fontweight);
    %legend('f', 'z');	
    %xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
    %ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
    hold off;

    % weight
    subplot 312;
    plot(1:train_delta:T, outData.Wo_norm, 'linewidth',linewidth);
    xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
    ylabel('|w|', 'fontsize', fontsize, 'fontweight', fontweight);
    
    subplot 313;
    plot(1:train_delta:T, outData.MSE, 'linewidth',linewidth);
    xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
    ylabel('MSE', 'fontsize', fontsize, 'fontweight', fontweight);
    pause(0.5);
  end
  
  if mod(ti,progress_interval) == 0
     fprintf('Progress : %d%%\n',floor(ti/T*100)); 
  end
  
end

fprintf('Training Completed, MAE : %f\n', mean(outData.MSE));

net.Wo = Wo;
net.Wr = Wr;
net.Wfb = Wfb;
net.B = B;

end

