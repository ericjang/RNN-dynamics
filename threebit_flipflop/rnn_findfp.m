% search for fixed points in an echo-state network
% given initial trajectory x0
function [fixed, history] = rnn_findfp(x0, net)
% THREEBIT_FINDFP - Numerically solves for a fixed point given echostate
% network weights and initial activation state x0.
% Returns: Nx1 fixed point that minimizes q(x) = .5|F(x)|
% and minimization trajectory along phase space.

Wr = net.Wr;
Wfb = net.Wfb;
Wo = net.Wo;
N = size(Wr,1);
B = net.B;
nIn = size(B,2);
u = zeros(nIn,1); % for the 3-bit flip-flop, long-term input = 0

assert(length(x0) == N, 'size of x0 should be a Nx1 vector');

% "global" variable that is accessible by optim_out_x
history.x = [];
history.q = [];
% inline optimfunc for recording trajectories through phase space
function stop = optim_out_x(x,optimValues,state)
   switch state
       case 'iter'
           history.x = [history.x x];
           history.q = [history.q optimValues.fval];
       otherwise
   end
   stop = false;
end % end optim_out_x

% inline function for RNN as defined in the 3-bit flip-flop task
function [q, grad, G] = rnn(x)
    [q,grad,G] = echostate_optim(x,u,net);
end

[fixed,fval,exitflag,output] = fminunc( @(x) rnn(x), x0, ...
              optimset('tolfun',1e-30,...
              'hessian','on','gradobj','on',...
              'display','off','OutputFcn', @optim_out_x) );


end
