% This script serves as the method to calculate the matrix M using the
% steepest gradient without the memory

 
clear all;
%% Import of raw data into matlab
raw_data = importdata('TIMIT39.mat');
d_train = raw_data.Speech_train;
d_output = raw_data.Target_train;

 

 
%% Parameter initialization
%square=15; % the square root of the length of the reservoir matrix
%N = square^2; % number of neurons in the reservoir-1
%phim=0; % phase factor for detuning
[d1, d2]=size(d_train); %size of the training data
T=800000; % cut-off length of the training data
%lambda=0.7^(3.5);
%scaling=2.5;
STEP=50000; % Transformation matrix iriteration steps
%b=normrnd(0,2,N,d1)*0.5^(scaling); % Matrix b, the reservoir weight matrix $W_{in}$
%b=normrnd(0,2,N,d1+1)*0.5^(scaling); %Matrix b, the reservoir weight matrix $W_{in}$
% Matrix a, the reservoir interconnection matrix $W_{res}$. Here we implement the function
% from the DOEcos2D_refl.m
%a=DOEcos2D_refl(N,phim);
%a=DOEcos2D_trans(N,phim);
%a=randn(N);
%a=a./max(abs(eig(a)));
%scaling_a=0;
%a=a.*scaling_a;% scaling of the matrix a 
%a=zeros(N); % the scaling of a equals to zero
 
eta0=0.001; % time step for the matri M's iterations
cut_int=1000; % cut interval of the array on the block for the calculation of M
x_train=zeros(d1+1,T); % The construction of the x matrix
c_err=zeros(1,STEP); % error array that records the input value
%x_test(N+1,:)=1;
%y_train=M*x_train; % The real output y matrix
%yy_test

 
%% Construction of the matrix x 
%x_train(1:N,1)=0;% all zero value in the first row of the x_test matrix
%u_train=d_output(:,1:T);
%phi =0; %phase factor 
% 
% % Input with bias
% % d_bias=0; % the added bias value 
% % d_bias_end=d_bias*ones(1,d2);
% % d_train=[d_train;d_bias_end];
% 
% % Normal a matrix
% 
% for j=1:T-1
%     x_train(1:N,j+1)=(a*x_train(1:N,j) + b*d_train(:,j)).*exp(1i*phi);
%     if mod(j,10000)==0
%         display2=['X MATRIX STEP: ',num2str(j)];
%         disp(display2);
%     end
% end
% 
% For the all-zero a matrix with no interaction
% for j=1:T-1
%     x_train(1:N,j+1)=(b*d_train(:,j)).*exp(1i*phi);
%     if mod(j,10000)==0
%         display2=['X MATRIX STEP: ',num2str(j)];
%         disp(display2);
%     end
% end
% 
x_train(1:d1,:)=d_train(:,1:T);
%d_bias=std(x_train(:));
%display_x_train=['Standard deviation of x_train is: ',num2str(d_bias)];
%disp(display_x_train);
% 
% x_train=abs(x_train+d_bias).^2;
%x_train=x_train+d_bias;
% 
x_train=(x_train-mean(x_train(:)))./std(x_train(:));
% 
  x_train(d1+1,:)=ones(1,T); % set the last line of the matrix x as all ones.



 
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
%M=zeros(d1, N+1); % The transformation matrix M
M=zeros(d1,d1+1); %The transformation matrix M for the no-memory case
%  MM=DOEcos2D_refl(N+1,phim);
%  M=MM(1:d1,:);% The transformation matrix M
%M_lsq=zeros(d1, N+1); % The transformation matrix M
%  MM_lsq=DOEcos2D_refl(N+1,phim);
%  M_lsq=MM_lsq(1:d1,:); % The transformation matrix M_lsq

 
k=1; % initial value for iriteration
eta0=0.00001; % time step for the matri M's iterations
eta0_lsq = 0.00002;
while k<=STEP
    eta = eta0*(1-k/STEP);
    eta_lsq = eta0_lsq*(1-k/STEP);

    
    k=k+1;

    
    rnd_cut = randi([1,T-cut_int],1,1);% Choose a random number between 1 and T-cut_int

    
    x_train_rnd=x_train(:,rnd_cut:rnd_cut+cut_int-1); %the cut-off x_train matrix
    y_train_rnd = M*x_train_rnd; % The real output y matrix y_train

    

    

    
    % Softmax function method:
    p_train_rnd = exp(y_train_rnd - repmat(max(y_train_rnd),39,1));
    p_train_rnd = p_train_rnd./repmat(sum(p_train_rnd),39,1);

    
    delta_y_rnd= p_train_rnd - (d_output(:,rnd_cut:rnd_cut+cut_int-1)+1)/2;

    
    grad_M = delta_y_rnd*x_train_rnd';
    class_error = 1 - sum(sum((d_output(:,rnd_cut:rnd_cut+cut_int-1)+1)/2.*(y_train_rnd==repmat(max(y_train_rnd),d1,1))))/size(y_train_rnd,2);% Calculate the classification error

 
%     y_train_rnd_lsq = M_lsq*x_train_rnd;
%     delta_y_rnd= y_train_rnd_lsq - d_output(:,rnd_cut:rnd_cut+cut_int-1);
%     grad_M_lsq = delta_y_rnd*x_train_rnd';
     M=M - eta*grad_M; % The iriteration process
%     M_lsq = M_lsq - eta_lsq*grad_M_lsq;

    
%     class_error_lsq = 1 - sum(sum((d_output(:,rnd_cut:rnd_cut+cut_int-1)+1)/2.*(y_train_rnd_lsq==repmat(max(y_train_rnd_lsq),d1,1))))/size(y_train_rnd,2);% Calculate the classification error
    c_err(1,k)=class_error; % record the value
%     c_err_lsq(1,k) = class_error_lsq;
    if mod(k,50)==0
        display3=['Random method step: ',num2str(k)];
        disp(display3); 
        
        %Plot the figure in realtime
%         plot(mean(reshape(c_err(1:k),50,k/50)),'r'); hold on;
%         plot(mean(reshape(c_err_lsq(1:k),50,k/50)),'b'); drawnow;
        plot(mean(reshape(c_err(1:k),50,k/50)),'r');drawnow;
        
        %Figure plot configuration
        xlabel('Step','Interpreter','LaTex');
        ylabel('Classification Error','Interpreter','LaTex');
        %titlename=['Classification Error with ','Bias=',num2str(d_bias),', Scaling=',num2str(scaling_a),', $\phi=$',num2str(phi),', $\phi_{m}=$',num2str(phim),', N=',num2str(N)];
        
        %leg1=legend('Softmax Fucntion','Matrix','Location','NorthEastOutside');
        leg1=legend('Softmax Function','Location','NorthEast');
        set(leg1,'FontAngle','italic','TextColor',[.3,.2,.1]);
        
    end

    

 
end

%% Save the results in image as eps in color
%filename1=['bias_',num2str(d_bias),'_scaling_a_', num2str(scaling_a)];
%filename2=['no_interaction_bias_',num2str(d_bias),'_N_',num2str(N)];
%fname='/Users/Tianqi/Documents/MATLAB/Reservoir Computing/bias';
%saves(gcf,fullfile(fname,filename1),'epsc');
%saveas(gcf,fullfile(fname, filename2),'epsc'); % save the image as eps format file
 
%% Test for the availability of the matrix M on the remaining data of x_train
t1=T+1;
t2=1010000;
dt=t2-t1+1;
% x_test=zeros(N+1,dt);
x_test=zeros(d1+1,dt);
u_test=d_train(:,t1:t2);
yy_test=d_output(:,t1:t2);% desired output data for the test
str_test='Now start to test on the new set of data.';
disp(str_test);
% for k=1:dt-1
%     x_test(1:N,k+1)=(a*x_test(1:N,k)+b*u_test(:,k)).*exp(1i*phi);
%     if mod(k,1000)==0
%         display_test=['Testing Step: ', num2str(k)];
%         disp(display_test);
%     end
% 
% end
%x_test_end=ones(1,dt);
%x_test=[x_test;x_test_end];

% x_test=abs(x_test+d_bias).^2;
x_test(1:d1,:)=u_test(:,:);
%d_bias_test=std(x_test(:));
%display_x_test=['Standard deviation of x_test is: ',num2str(d_bias_test)];
%disp(display_x_test);
%x_test=x_test+d_bias_test;
x_test=(x_test-mean(x_test(:)))./std(x_test(:));
x_test(d1+1,:)=ones(1,dt); % set the last line of the matrix x as all ones

% Calculate the real output data
y_test=M*x_test;


% Calculate the error rate of the data
class_error_test = 1 - sum(sum((yy_test+1)/2.*(y_test==repmat(max(y_test),d1,1))))/size(y_test,2);
display_result=['The Classification Error is: ',num2str(class_error_test)];
disp(display_result);

% Figure title and record the error rate of the data in it

titlename=['No-memory classification error with test error: ',num2str(class_error_test),', $\eta=$',num2str(eta0)];
title(titlename,'Interpreter','LaTex');











