function [ outData ] = test_rnn( input, params, net)
% tests echostate network with provided readout weights

    T = params.T; % number of timesteps
    tau = net.tau; % slowdown
    act_fun = net.act_fun; % activation nonlinearity
    Wr = net.Wr;
    Wfb = net.Wfb;
    Wo = net.Wo;
    B = net.B;

    % Initialize Network 
    nOut = size(Wo,1);
    N = size(Wr,1);

    % initial state
    if isfield(net,'x0')
        x = net.x0; % user-supplied
    else 
        x = 0.5*randn(N,1);
    end

    outData.Z = zeros(nOut,T);
    if params.recordX
        outData.X = zeros(N,T);
        outData.X(:,1)=x;
    end
    if params.recordQ
        outData.Q = zeros(1,T);
    end
    
    if params.recordR
        outData.R = zeros(N,T);
    end

%    progress_interval = ceil(T/100);



    for ti=2:T
          r = act_fun(x);
          z = Wo*r;
          u = input(:,ti);
          Bu = B*u;
          xdot = -x+Wr*r+Wfb*z+Bu;
          x = x + xdot/tau; % integrate with time constant
          outData.Z(:,ti)=z;
          if params.recordX
              outData.X(:,ti) = x;
          end
          if params.recordQ
             F = xdot - Bu;
             outData.Q(ti) = 0.5*(sum(F.^2)); 
          end
          if params.recordR
             outData.R(:,ti) = r; 
          end
%           if mod(ti,progress_interval) == 0
%             fprintf('Progress : %d%%\n',floor(ti/T*100)); 
%           end
    end

end
