function [ modes, motion ] = gen_stim_motor( T, P_wr, P_rw )
%GEN_STIM generates mode-switching behavior in an echo-state network
% data retrieved from 
% http://mocap.cs.cmu.edu/search.php?subjectnumber=16
% P_wr = probability of switching from walk to run
% P_rw = probability of switching from run to walk
% T = number of frames
% mode-switching is done via Markov chain with bernoulli transition
% probabilities

% 16_32 - walk
% 16_35 - run/jog

D = load('motions.mat');
motion_walk = D.walk';
% truncate the beginning of walk - it's not cyclic
motion_walk = motion_walk(:,100:end);
motion_run = D.run';

assert(size(motion_walk,1) == size(motion_run,1), 'mocap data DOFs are mismatched');

dof = size(motion_walk,1);

motion = zeros(dof,T);
modes = zeros(1,T);

i_walk = 1;
i_run = 1;
m = 1; % start out walking 
for i=1:T
    modes(i) = m;
    if m == 1 % walking
        motion(:,i) = motion_walk(:,i_walk);
        i_walk = mod(i_walk, size(motion_walk,2)) + 1;
        if rand() < P_wr
            m = 2;
        end
    elseif m == 2 % running
        motion(:,i) = motion_run(:,i_run);
        i_run = mod(i_run,size(motion_run,2)) + 1;
        if rand() < P_rw
            m = 1;
        end
    end
end

end

