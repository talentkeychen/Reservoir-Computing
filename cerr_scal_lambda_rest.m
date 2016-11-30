%% Parameter initilization
x=2;% dimension of the lambda
y=3;% dimension of the weight matrix
lambda=zeros(x,1);
scaling=zeros(y,1);
c_err=zeros(x,y); % clear error matrix that records the incoming value
% Construction of the lambda value matrix
for k=1:x
    lambda(k)=0.7^(k-1);
end
% Construction of the scaling value matrix
for l=1:y
    scaling(l)=l+4;
end


%% Iriteration steps of trials and recordings of the values
str5='Error recording case: ';
for i=1:x
    for j=1:y
        c_err(i,j)=clear_error(lambda(i),scaling(j));
        display5=[str5, num2str(i), num2str(j)];
        disp(display5);
    end
end

%% Plot of the input neuron state signal
%figure, plot(x_train(10,:));

%% Plot of the clear error as a function of $\lambda$ and the scaling of the weight matrix Parameter initilization
figure, surf(c_err);
