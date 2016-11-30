function [ class_error ] = clear_error( lambda, scaling )
%CLEAR_ERROR Calculate the clear error with respect to different input
%scalings and $\lambda$, the parameter of the low-pass filter
%   Detailed explanation goes here.

%   'You Shall Not Pass!' by Gandulf the White, Quoted from 'Lord of the Rings'.

%% Import of raw data into matlab
raw_data = importdata('TIMIT39.mat');
d_train = raw_data.Speech_train;
d_output = raw_data.Target_train;

%% Parameter initialization
N = 200; % number of neurons in the reservoir
step = 1000; %number of iriteration
%remain = 0; % cut of some data for test
Nout=39; % number of output channel
%lambda=0.3; % parameter of the low-pass filter
[d1, d2]=size(d_train);
T=800; % number of the cut-off interval, which means we hereby choose 1000*1000=1000000 original data points
%T=max(size(d_train))-remain;
%ratio_data=zeros(step,1); % the vector valumn to record the ration values.
b=normrnd(0,2,N,d1)*0.5^(scaling); % Matrix b, the reservoir weight matrix $W_{in}$
% Matrix a, the reservoir interconnection matrix $W_{res}$ 
a=randn(N); 
a=a./max(abs(eig(a)));

%a=zeros(N);%delayed feedback
%for j=2:N
%    a(j,j-1)=1;
%end
%a(1,N)=1;
A=zeros(N+1,N+1);% Matrix A
B=zeros(N+1,d1);% Matrix B

%% Calculation of the matrix x
% First round with all-zero column section
x_train_tmp=zeros(N,T);
x_train_tmp(:,1)=zeros(N,1);
u_train=d_train;
y_output=d_output(:,1:800)';
for j=1:T-1
    x_train_tmp(:,j+1)=(1-lambda)*x_train_tmp(:,j)+lambda*tanh(a*x_train_tmp(:,j)+b*u_train(:,j));
end
x_train_end=ones(1,T);
x_train=[x_train_tmp; x_train_end];
A=A+x_train*x_train';
B=B+x_train*y_output;
% The rest round of iteration
str1='Step 1: Training the reservoir using ';
str2=' neurons.';
str3='Step: ';
display1=[str1,num2str(N),str2];
disp(display1);
for i=2:step
    %b=b*0.5
    %% Train the reservoir
    %x_0_train=zeros(N,1);
    x_train_tmp(:,1)=tanh(a*x_train_tmp(:,T)+b*u_train(:,T));
    u_train=d_train(:,(i-1)*800+1:i*800);
    y_output=d_output(:,(i-1)*800+1:i*800)';
    %x_train_tmp=zeros(N,T);
    
    % Reciprocal formula
    %x_train_tmp(:,1)=x_0_train;
    for j=1:T-1
       x_train_tmp(:,j+1)=(1-lambda)*x_train_tmp(:,j)+lambda*tanh(a*x_train_tmp(:,j)+b*u_train(:,j)); 
    end
    x_last_train=ones(1,T);
    x_train=[x_train_tmp;x_last_train];
    A=A+x_train*x_train';
    B=B+x_train*y_output;
    if mod(i,100)==0
        display2=[str3, num2str(i)];
        disp(display2);
    end
    
end

%% Optimization of matrix M
M=(A\B)';

%% Test on the new set of data from the remaining training data
% Choose the input test data from time step
% 800001 to 1010000
% x_test=d_train(:,1000001:1010000);
t1=800001;
t2=1010000;
dt=t2-t1+1;
x_test_tmp=zeros(N,t2-t1+1);
u_test=d_train(:,t1:t2);
yy_test=d_output(:,t1:t2);% desired output data for the test
str4='Now start to test on the new set of data.';
disp(str4);
for k=1:dt-1
    x_test_tmp(:,k+1)=(1-lambda)*x_test_tmp(:,k)+lambda*tanh(a*x_test_tmp(:,k)+b*u_test(:,k));
    if mod(k,1000)==0
        display2=[str3, num2str(k)];
        disp(display2);
    end

end
x_test_end=ones(1,dt);
x_test=[x_test_tmp;x_test_end];
% Calculate the real output data
y_test=M*x_test;

%% Calculate the error rate of the data
class_error = 1 - sum(sum((yy_test+1)/2.*(y_test==repmat(max(y_test),Nout,1))))/size(y_test,2);

%% Plot of the input neuron state signal
%figure, plot(x_train(10,:));

end

