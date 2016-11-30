
%% Generation of training and testing dataset
d_train=dataset_narma_10(1,1024,1);
d_test=dataset_narma_10(1,1024,1);

%% Parameter initialization
N=100; %number of neurons
step=25; % time step of the iriteration
T=1024; % number of steps of time
ratio_data=zeros(step,1); % the vector column to record values of ratios
b=zeros(N); % Matrix b
%a=randn(N); 
%a=a./max(abs(eig(a)));
%% Contruction of martix a using delayed feedback
a=zeros(N);
for j=2:N
    a(j,j-1)=1;
end
a(1,N)=1;
%%

%% Calculation of ratio
for i=1:step
    %% Different reservoir weight matrix
    %b=normrnd(0,2,N,1)*0.1^(step-i); % alternate the weight by the prefactor 0.1 to the power of step-i
    %b=normrnd(0,2,N,1)*0.5^(step-i); % alternate the weight by the prefactor 0.5 to the power of step-i
    b=normrnd(0,2,N,1)*0.5^(i); %direct to the power of 0.5
    %b=normrnd(0,2,N,1)*i;
    %b=normrnd(0,i,N,1); % alternate the weight by std. of the normal randoms
    
    %% Training the reservoir
    x_0_train=zeros(N,1);
    x_0_train(N)=1;
    % External Inputs
    u_train=d_train.inputs;
    % Matrix x
    x_train=zeros(N,T);

    % Reciprocal formula
    x_train(:,1)=x_0_train;
    for n=1:T-1
        x_train(:,n+1)=tanh(a*x_train(:,n)+b*u_train(n));
    end
    x_train(N,:)=1;

    %Regression Process
    A=x_train*x_train'+0.001*eye(N);
    %A=x_train*x_train';
    yy_train=d_train.outputs; % desired outputs y*
    B=x_train*yy_train';
    M=A\B; %Solve the linear equation of Ax=B
    MM=M'; % transportation of M
    %% Test on the new set of data
    x_0_test=zeros(N,1);
    x_0_test(N)=1;
    %External Inputs
    u_test=d_test.inputs;
    %Matrix x
    x_test=zeros(N,T);
    % Reciprocal formula
    x_test(:,1)=x_0_test;
    for n=1:T-1
        x_test(:,n+1)=tanh(a*x_test(:,n)+b*u_test(n));
    end
    x_test(N,:)=1;
    y_test=MM*x_test;% real output
    yy_test=d_test.outputs;% desired output
    delta_test=yy_test-y_test;
    v_test=var(yy_test-y_test); %variance of errors
    v_desired=var(yy_test);
    ratio=v_test/v_desired; % calculate the ratio
    ratio_data(i,1)=ratio; % record the value of ratio in the vector
end

%% Plot the ratio data trajectory
figure, plot(ratio_data,'r*');
%xlabel('Test number (Step)');
xlabel('$ln(W_{in})/ln(0.5)$','Interpreter','LaTex');
ylabel('$Radio$ $Value$','Interpreter','LaTex');
title('Different ratios as a function of prefactor 0.5, with 0.001 regularization');



