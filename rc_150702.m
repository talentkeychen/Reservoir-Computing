
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                        %
%Reservoir Computing Practice 2015-07-02 %
%                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initial data generation
dd=dataset_narma_10(1,1024,1);

%% Generate the initial state x(1)
N=100; %number of neurons
x_0=zeros(N,1);
%for i=1:N-1
 %   rr=randn(1);
  %  if rr<0
   %     x_0(i)=0
    %else
     %   x_0(i)=1
    %end 
%end
x_0(N)=1;

%% External Inputs
u=dd.inputs;
%% Matrix b
b=zeros(N);
b=normrnd(0,2,N,1);
%% Matrix a 
a=randn(N);
a=a./max(abs(eig(a)));
%% Matrix x
T=1024; % numbers of steps of time
x=zeros(N,T);

%% Reciprocal formula
x(:,1)=x_0;
for n=1:T-1
    x(:,n+1)=tanh(a*x(:,n)+b*u(n));
end
x(N,:)=1;

%% Regression Process
A=x*x';
yy=dd.outputs; % desired outputs y*
B=x*yy';
M=A\B; %Solve the linear equation of Ax=B

%% Histogram of resevoir states
figure, hist(x(:))

%% Performance checking
MM=M';
y=MM*x;
delta_y=y-yy; % displacement of real and desired outputs
v=var(delta_y);
dd=delta_y.^2;
vv=mean(dd);








