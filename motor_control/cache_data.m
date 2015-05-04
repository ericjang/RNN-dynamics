% walk, run = numFrames x degrees of freedom matrix
walk = amc_to_matrix('16_32.amc');
run = amc_to_matrix('16_35.amc');
save('motions.mat','walk','run');