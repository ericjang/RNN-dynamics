% plot evolution of phase space as a function
% of training iterations
clear;close all;
apply_settings;

fpname = sprintf('data/FPs_%s.mat',prefix);
load(fpname); % for the fixed points, W, mu, we

fname = sprintf('data/%s_train_progression_pca.mat',prefix);
load(fname);

train_fname = sprintf('data/%s_train.mat',prefix);
load(train_fname); % net

Wo_interval=p.Wo_interval;

data = FPs';
[W,Yfp] = pca(data,'VariableWeights','variance','Centered',true); % data : rows = observations, columns = variables
[nWos, nICs] = size(trajectories2_pca);

for i=1:nWos % for each weight progress
   %subplot(1,nWos,i);
   fprintf('%d\n',i);
   figure;
   hold on;
   set(gcf, 'Visible', 'off')
   
   % plot fixed points
   scatter3(Yfp(:,1),Yfp(:,2),Yfp(:,3),'go'); 
   
   % plot test IC
   Y = trajectories_pca{i};
   plot3(Y(:,1),Y(:,2),Y(:,3),'b');
   
   xlim([-60,60]);
   ylim([-60,60]);
   zlim([-60,60]);
   
   
   % random IC
   for j=1:nICs
       Y = trajectories2_pca{i,j};
       plot3(Y(:,1),Y(:,2),Y(:,3),'r');
   end
   msg = sprintf('Iter : %d', i*Wo_interval);
   title(msg);
   
   set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 4])
   
   az = 45;
   el = 25;
   view(az, el);
   
   fname = sprintf('images/fp_evolve_%s/%.2d.png',prefix,i);
   print(fname,'-dpng','-r100'); 
end
