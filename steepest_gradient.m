% This script serves as the method to calculate the matrix M using the
% steepest gradient.

clear all;
%% Import of raw data into matlab
raw_data = importdata('TIMIT39.mat');
d_train = raw_data.Speech_train;
d_output = raw_data.Target_train;


%% Parameter initialization
N = 200; % number of neurons in the reservoir
[d1, d2]=size(d_train); %size of the training data
T=800000; % cut-off length of the training data
lambda=0.7^(4);
scaling=5; % scaling factor
STEP=1000; % Transformation matrix iriteration steps
b=normrnd(0,2,N,d1)*0.5^(scaling); % Matrix b, the reservoir weight matrix $W_{in}$
% Matrix a, the reservoir interconnection matrix $W_{res}$ 
a=randn(N); 
a=a./max(abs(eig(a)));
M=zeros(d1, N+1); % The transformation matrix M
eta=0.000001; % time step for the matri M's iterations
cut_int=1000; % cut interval of the array on the block for the calculation of M
x_train=zeros(N+1,T); % The construction of the x matrix
c_err=zeros(1,STEP); % error array that records the input value
%x_test(N+1,:)=1;
%y_train=M*x_train; % The real output y matrix
%yy_test

%% Construction of the matrix x 
x_train(1:N,1)=0;% all zero value in the first row of the x_test matrix
u_train=d_output(:,1:T);
for j=1:T-1
    x_train(1:N,j+1)=(1-lambda)*x_train(1:N,j)+lambda*tanh(a*x_train(1:N,j)+b*u_train(:,j));
    if mod(j,10000)==0
        display2=['X MATRIX STEP: ',num2str(j)];
        disp(display2);
    end
end
    x_train(N+1,:)=ones(1,T); % set the last line of the matrix x as all ones.

%% Steepest gradient iriteration
%  Direct matrix method:
% for i=1:STEP
%     str1='Transformation Step: ';
%     display1=[str1,num2str(i)];
%     disp(display1);
%     y_train=M*x_train; % The real output y matrix y_train
%     yy_train=d_output(:,1:T); % The desired output matrix yy_train 
%     delta_y=y_train-yy_train;
%     grad_M=delta_y*x_train'; % Calculate the gradient matrix
%     M=M-eta*grad_M; % The iriteration process
% end
    
%  Random matrix part choosing method:
k=1; % initial value for iriteration
while k<=STEP
    rnd_cut=randi([1,T-cut_int],1,1);% Choose a random number between 1 and T-cut_int
    x_train_rnd=x_train(:,rnd_cut:rnd_cut+cut_int-1);
    y_train=M*x_train_rnd; % The real output y matrix y_train
    delta_y_rnd=y_train-d_output(:,rnd_cut:rnd_cut+cut_int-1); %the random-chosen cut-off delta_y matrix
    %x_train_rnd=x_train(:,rnd_cut:rnd_cut+cut_int-1); %the cut-off x_train matrix
    grad_M=delta_y_rnd*x_train_rnd';
    M=M-eta*grad_M; % The iriteration process
    k=k+1;
    class_error = 1 - sum(sum((d_output(:,rnd_cut:rnd_cut+cut_int-1)+1)/2.*(y_train==repmat(max(y_train),d1,1))))/size(y_train,2);% Calculate the classification error
    c_err(1,k)=class_error; % record the value
    if mod(k,50)==0
        display3=['Random method step: ',num2str(k)];
        disp(display3);
    end
    plot(c_err(1:k));drawnow;
 % Softmax function method:
    




end

%% Test on the availability of the matrix M










