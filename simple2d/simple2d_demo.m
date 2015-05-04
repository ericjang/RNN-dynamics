function [] = simple2d_demo

clear;
close all;

xmin = -1.5;
xmax = 1.5;

x1dot = @(x1,x2) (1-x1.^2).*x2;
x2dot = @(x1,x2) x1./2-x2;
q = @(x1,x2) 0.5*(x1dot(x1,x2).^2 + x2dot(x1,x2).^2);
xx = xmin:0.02:xmax;
[x1s,x2s] = meshgrid(xx,xx);

fprintf('Plotting Energy q(X)...\n');
figure(1);
Z = q(x1s,x2s);
num_levels = 20;
contourf(x1s,x2s,Z,num_levels); 
axis square;
colorbar;


% overlay coarser quiver plot on top
fprintf('Plotting Vector Field F(X)...\n');
hold on;
xx = xx(1:10:end);
[x1s,x2s] = meshgrid(xx,xx);
dx1 = x1dot(x1s,x2s);
dx2 = x2dot(x1s,x2s);
quiver(x1s,x2s,dx1,dx2,'w');
xlim([xmin, xmax]);
ylim([xmin, xmax]);
axis square;
xlabel('x');
ylabel('y');
%title('Vector Field F(x)');


% q-optimization
fprintf('Running q-optimization...\n');

% "global" variable that is accessible by optim_out_x
history.x = [];


    % define inline OutputFcn for recording trajectory
    % see http://www.mathworks.com/help/optim/ug/output-functions.html#brjhnpu
    % for more full-featured example
    function stop = optim_out_x(x,optimValues,state)
       stop = false;
       switch state
           case 'iter'
               history.x = [history.x x];
           otherwise
       end
    end % end optim_out_x 

num_trajectories = 10;
for i=1:num_trajectories
    history.x = []; % clear it
    xstart = xmin + rand(2,1)*(xmax-xmin);
    [fixed,fval,exitflag,output] = fminunc( @(x) simple2d(x), xstart, ...
                  optimset('tolfun',1e-20,...
                  'hessian','on','gradobj','on',...
                  'display','off','OutputFcn', @optim_out_x) );
    plot(fixed(1),fixed(2),'.r','linewidth',3,'markersize',15);
    
    % jacobian
    x=fixed(1);
    y=fixed(2);
    J = [-2*x*y, 1-x^2; 1/2,  -1]; 
    [v d] = eig(J);
    [x,y]
    fprintf('lambda_1 = %f\n',v(1));
    fprintf('lambda_2 = %f\n',v(2));
    % plots the history
    %plot(history.x(1,:),history.x(2,:),'k','LineWidth',3);
end

end






% [fixed val eFlag]= fminunc( @(z) g(z(1),z(2)),[0 0],optimset('tolfun',1e-8,'hessian','off','gradobj','off','display','off') );
% plot(fixed(1),fixed(2),'.r','linewidth',3,'markersize',15)
% [fixed val eFlag]= fminunc( @(z) g(z(1),z(2)),[5 5],optimset('tolfun',1e-8,'hessian','off','gradobj','off','display','off') );
% plot(fixed(1),fixed(2),'.r','linewidth',3,'markersize',15)
% J=[-2*fixed(1) 1;1 -1];
% [v d] = eig(J);