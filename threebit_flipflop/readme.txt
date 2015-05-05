THREEBIT_FLIPFLOP

The code in this folder implements an echo-state network (ESN) that is trained
to perform the 3-bit flip/flop task. FORCE-style training is done using the Recursive
Least Squares (RLS) algorithm on the readout weights Wo, which is equivalently a 
rank-3 modification to the combined weight matrix W_c = W+W_fb*Wo

Here is the general pipeline for running these code examples, in numbered order.

1) run_threebit_train.m 
    OUTPUTS : 
        net3.mat - ( network parameters & trajectory from training & weight evolution)
        net3_test.mat - (single test trajectory)

2) run_qoptim.m
    INPUTS : 
        net3.mat - from (1)
        net3_test.mat - from (1)
    OUTPUTS : 
         q3.mat - fixed points in the system, as well as trajectories 
                  from ICs and randomly sampled phase space
    Note: between re-runs of run_threebit_train, this script will need to be re-run.
    That is because the principal components for any given RNN will be unlikely to match up.

3) run_pca_progress.m 
    INPUTS:
        net3.mat - 
        q3.mat -   
    OUTPUTS:
        train_progression_pca.mat - PCA-flattened test&random trajectories for each Wo snapshot

4) analyze_train_progress.m
    INPUTS:
        training_progression.mat
        train_progression_pca.mat

[Optional Pipeline]:
2) net3.mat > structural_stability.m 
takes net3.mat computes MSE of training trajectory as a function of perturbation.

2) analyze_train_stability.m
  plots Test MSE as a function of training iteration

process_train_progress
