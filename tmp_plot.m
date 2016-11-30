% This script plots the classification error of the test data for different
% learning rates as well as different numbers of previous history columns.

eta=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3];
pre_3=[.44022, .42356, .41913, .41538, .41327, .41272];
pre_4=[.43631, .42111, .41627, .41241, .41027, .41153];
pre_5=[.4329, .419, .41433, .41133, .4091, .41092];
pre_10=[.42754, .41404, .41088, .4084, .40755, .41464];
his_futr=[.41566, .4051, .40204, .40049, .40225, .41053];
no_mem=[.4772, .4547, .4477, .44214, .43759, .43634];

figure, plot(eta,pre_3,'ro-');hold on;
plot(eta,pre_4,'go-');hold on;
plot(eta,pre_5,'bo-');hold on;
plot(eta,pre_10,'co-');hold on;
plot(eta,his_futr,'mo-');hold on;
plot(eta, no_mem,'yo-');
xlabel('Learning Rate $\eta$','Interpreter','LaTex');
ylabel('Test Error','Interpreter','LaTex');
leg1=legend('Previous Three Columns','Previous Four Columns', 'Previous Five Columns','Previous Ten Columns','Five History and Future Columns','No Memory', 'Location','NorthEast');
set(leg1,'FontAngle','italic','TextColor',[.3,.2,.1]);
titlename=('Different Test Error as a function of Learning Rate $\eta$');
title(titlename,'Interpreter','LaTex');

% eta_no=[1e-6, 1e-5, 1e-4, 5e-4, 75e-5, 1e-3, 15e-4, 25e-4, 5e-3, 1e-2];
% no_mem=[.52753, .477, .44815, .4377, .43712, .43663, .43578, .43625, .43906, .44978];
% 
% figure, plot(eta_no, no_mem, 'ro-');
% xlabel('Learning Rate $\eta$','Interpreter','LaTex');
% ylabel('Test Error','Interpreter','LaTex');
% %leg1=legend('Previous Three Columns','Previous Four Columns', 'Previous Five Columns', 'Location','NorthEast');
% %set(leg1,'FontAngle','italic','TextColor',[.3,.2,.1]);
% titlename=('Different Test Error as function of Learning Rate $\eta$ for the no-memory case');
% title(titlename,'Interpreter','LaTex');
