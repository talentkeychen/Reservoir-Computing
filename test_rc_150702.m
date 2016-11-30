d_test=dataset_narma_10(1,1024,1);

%% Generate the initial state x(1)

x_0_test=zeros(N,1);
%for i=1:N-1
 %   rr=randn(1);
  %  if rr<0
   %     x_0(i)=0
    %else
     %   x_0(i)=1
    %end 
%end
x_0_test(N)=1;

%% External Inputs
u_test=d_test.inputs;

%% Matrix x
T=1024; % numbers of steps of time
x_test=zeros(N,T);

%% Reciprocal formula
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
ratio=v_test/v_desired;

