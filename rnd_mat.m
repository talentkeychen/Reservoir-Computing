c_err=zeros(1,STEP);
eta=0.0001;
k=1; % initial value for iriteration
cut_int=50; % cut interval of the array on the block for the calculation of M
while k<=STEP
    rnd_cut=randi([1,T-cut_int],1,1);% Choose a random number between 1 and T-cut_int
    y_train=M*x_train; % The real output y matrix y_train
    delta_y_rnd=y_train(:,rnd_cut:rnd_cut+cut_int-1)-d_output(:,rnd_cut:rnd_cut+cut_int-1); %the random-chosen cut-off delta_y matrix
    x_train_rnd=x_train(:,rnd_cut:rnd_cut+cut_int-1); %the cut-off x_train matrix
    grad_M=delta_y_rnd*x_train_rnd';
    M=M-eta*grad_M; % The iriteration process
    k=k+1;
    class_error = 1 - sum(sum((d_output(:,1:T)+1)/2.*(y_train==repmat(max(y_train),d1,1))))/size(y_train,2);% Calculate the classification error
    c_err(1,k)=class_error; % record the value
    if mod(k,50)==0
        display3=['Random method step: ',num2str(k)];
        disp(display3);
    end
    plot(c_err(1:k));drawnow;
 % Softmax function method:
    




end