function [ mat_out ] = lnr( mat_in )
%This function lnr serves as the way to shift and concatenate the matrix
%each five sub-column in a column of the new matrix
%% Parameters
[d1,d2]=size(mat_in);
mat_out=zeros(d1*11,d2-10);%parameter initilization 

%mat_out=zeros(d1*3,d2-2);
%%
mat_in_2=mat_in(:,1:d2-1);
mat_in_3=mat_in(:,1:d2-2);
mat_in_4=mat_in(:,1:d2-3);
mat_in_5=mat_in(:,1:d2-4);
mat_in_6=mat_in(:,1:d2-5);
mat_in_7=mat_in(:,1:d2-6);
mat_in_8=mat_in(:,1:d2-7);
mat_in_9=mat_in(:,1:d2-8);
mat_in_10=mat_in(:,1:d2-9);
mat_in_11=mat_in(:,1:d2-10);

mat_out(1:d1,:)=mat_in(:,11:d2);
mat_out(d1+1:2*d1,:)=mat_in_2(:,10:d2-1);
mat_out(2*d1+1:3*d1,:)=mat_in_3(:,9:d2-2);
mat_out(3*d1+1:4*d1,:)=mat_in_4(:,8:d2-3);
mat_out(4*d1+1:5*d1,:)=mat_in_5(:,7:d2-4);
mat_out(5*d1+1:6*d1,:)=mat_in_6(:,6:d2-5);
mat_out(6*d1+1:7*d1,:)=mat_in_7(:,5:d2-6);
mat_out(7*d1+1:8*d1,:)=mat_in_8(:,4:d2-7);
mat_out(8*d1+1:9*d1,:)=mat_in_9(:,3:d2-8);
mat_out(9*d1+1:10*d1,:)=mat_in_10(:,2:d2-9);
mat_out(10*d1+1:11*d1,:)=mat_in_11(:,1:d2-10);


%  mat_out(1:d1,:)=mat_in(:,3:d2);
%  mat_out(d1+1:2*d1,:)=mat_in_2(:,2:d2-1);
%  mat_out(2*d1+1:3*d1,:)=mat_in_3(:,1:d2-2);


end

