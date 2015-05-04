function [ u, target ] = gen_stim( T, nOut )
%GEN_STIM Generates input and corresponding target sequence
% for the 3-bit flip flop task.
%   T = nsteps
%   nOut = number of channels

pulse_duration = 10; % number of time steps each pulse lasts
u = zeros(nOut,T);

nPulses = floor(T*0.02); 
tstart = randperm(T,nPulses); % spike times
for i=1:nPulses
    if tstart(i) <= T-pulse_duration
        % one of the channels spike
        u(randi(nOut),tstart(i):tstart(i)+pulse_duration) = 2*(rand() < .5)-1;
    end
end




target = zeros(nOut,T);

% compute target function
prev = zeros(nOut,1);
for t=1:T
  changed = find(u(:,t) ~= 0);
  prev(changed) = u(changed,t);
  target(:,t) = prev;
end


end

