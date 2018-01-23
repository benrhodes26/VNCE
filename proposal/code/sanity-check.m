%----------------------------
% Simple mixture of Gaussians
%----------------------------

% mixing proportions: 0.5 and 0.5
% mean0 = mean1 = 0
% sigma1 = 1
% sigma0: parameter of interest


sigma1 = 1;
sigma0 = 4;

% number of data points
n = 100000;

%% generate observed data

xobs = zeros(1,n);
z = rand(1,n)>0.5;

x = (z==0).*(randn(1,n)*sigma0) + (z==1).*(randn(1,n)*sigma1);

% check
u = [-10:0.01:10];
figure
ksdensity(x,u)
hold on

px = 0.5*(normpdf(u,0,sigma0)+normpdf(u,0,sigma1));

plot(u,px,'r')
xlabel('u')
ylabel('pdf')
legend('kde','true')
grid

%% NCE objective function 

nu = 1;

theta = [0.01:0.1:8];
% correctly normalised model
pm = @(u, theta)( 0.5*(normpdf(u,0,theta)+normpdf(u,0,sigma1)) );

pn = @(u) ( normpdf(u,0,sigma0) ); % noise that matches the larger std

h = @(u,theta) ( log(pm(u,theta)./pn(u)) ); % quick hack: generally better to work in log domain!

y = sigma0*randn(1,n*nu);

nTheta = length(theta);
J = zeros(1,nTheta);
for k=1:nTheta
    thetak = theta(k);
    J(k) = -mean( log(1+nu*exp(-h(x,thetak))) ) - nu* mean( log(1+1/nu*exp(h(y,thetak))) );
end

    
figure
plot(theta,J)
hold on
plot(sigma0*[1 1],get(gca,'ylim'),'k--')
xlabel('parameter')
ylabel('objective')

[val,index]=max(J);
plot(theta(index),J(index),'ko')
legend('NCE objective','true parameter value','maximum of NCE obj')

%% First lower bound

q0 = @(x) ( 1./(1+sigma0/sigma1*exp(-(x.^2)*(1/sigma1^2-1/sigma0^2))) );
q1 = @(x) ( 1-q0(x) );

phi0 = @(u,theta) ( 0.5*normpdf(u,0,theta) );
phi1 = @(u,theta) ( 0.5*normpdf(u,0,sigma1) );

r0 = @(u,theta) (  phi0(u,theta)./( q0(u).*pn(u)+eps ) );
r1 = @(u,theta) (  phi1(u,theta)./( q1(u).*pn(u)+eps ) );

zx = rand(1,n)<q1(x);

J1 = zeros(1,nTheta);
Jlower = zeros(1,nTheta);
for k=1:nTheta
    thetak = theta(k);
    
    % compute expectation 
    m = q0(y).*r0(y,thetak) + q1(y).*r1(y,thetak);
    
    % J1 objective
    J1(k) = -mean( log(1+nu*1./r(x,zx,thetak)))- nu* mean( log(1+1/nu*m) );    
    
    % Jlower is the objective after inserting the bound for the first term in the NCE obj.
    %Jlower(k) = -mean( log(1+nu*1./r(x,zx,thetak)))- nu* mean( log(1+1/nu*exp(h(y,thetak))) );
    
end


figure
plot(theta,J)
hold on
plot(theta,J1,'r')

[val1,index1]=max(J1);

% true param value
plot(sigma0*[1 1],get(gca,'ylim'),'k--')

% maximiser of NCE obj
plot(theta(index),J(index),'ko')

% maximiser of lower bound
plot(theta(index1),J1(index1),'rs')

xlabel('parameter')
ylabel('objective')

legend('NCE objective','Lower bound','true parameter value','maximum of NCE obj','maximum of lower bound')

