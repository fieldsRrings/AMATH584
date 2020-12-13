clc;
close all;

% I tried using a freely available function to download and open the test and 
% training MNIST dataset but I couldn't get them to work so I just found .csv 
% versions of them online and used those. I hope that's okay. 

te_A = load('test_images.csv'); %loads corresponding files into arrays.
te_b = load('test_labels.csv');
tr_A = load('train_images.csv');
tr_b = load('train_labels.csv');

%% Training

x1_tr = tr_A\tr_b; %solves Ax=b with backslash.

x2_tr = pinv(tr_A)*tr_b; %solves Ax=b with pseudo inverse. 

x3_tr = lasso(tr_A,tr_b,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.

x3a_tr = lasso(tr_A,tr_b,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.

x4_tr = robustfit(tr_A,tr_b); %robust fit for Ax=b.

x5_tr = ridge(tr_b,tr_A,0.01,0); %ridge for Ax=b. I used 0.01 for k. 

%% Adjust Size from Robustfit and Ridge.
%Robustfit and Ridge kept returning a 785x1 vector. I just
%took off the last entry since it was zero for both functions. Not sure
%how else to fix it. You can't adjust iterations in either of them.

x4_tr = x4_tr(1:end-1);
x5_tr = x5_tr(1:end-1);

%% Plots for Training Data

subplot(3,2,1)
plot(x1_tr)
title('Backslash')
subplot(3,2,2)
plot(x2_tr)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x3_tr)
title('Lasso Lambda')
subplot(3,2,4)
plot(x3a_tr) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x4_tr)
title('Robustfit')
subplot(3,2,6)
plot(x5_tr)
title('Ridge')

%% Error 

E1_tr =  norm(te_b-te_A*x1_tr)/norm(te_b); %backslash
E2_tr =  norm(te_b-te_A*x2_tr)/norm(te_b); %pseudo
E3_tr =  norm(te_b-te_A*x3_tr)/norm(te_b); %Lasso with Lambda
E3a_tr =  norm(te_b-te_A*x3a_tr)/norm(te_b); %Lasso with Lambda and Alpha
E4_tr =  norm(te_b-te_A*x4_tr)/norm(te_b); %Robustfit
E5_tr =  norm(te_b-te_A*x5_tr)/norm(te_b); %Ridge

%% Error Bar Graph
%A*x=b
%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_tr E2_tr E3_tr E3a_tr E4_tr E5_tr];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error')

%% Testing Vectors using Training Data from methods.
%Uses test matrix and map x from training data to generate outcome vector b.
b1_te = te_A*x1_tr; %backslash
b2_te = te_A*x2_tr; %pseudo
b3_te = te_A*x3_tr; %Lasso with 1 norm.
b3a_te = te_A*x3a_tr; %Lasso with 1 norm and 2 norm.
b4_te = te_A*x4_tr; %Robustfit
b5_te = te_A*x5_tr; %Ridge

%% Plot of Actual outcome vs. Test outcomes.

t = [1:10];
plot(t,te_b(1:10),t,b1_te(1:10),t,b2_te(1:10),t,b3_te(1:10),t,b3a_te(1:10),t,b4_te(1:10),t,b5_te(1:10));
ylim([-2 10])
title('Test vs Train Methods(First 10 Points)')
legend({'True','Backslash','Pseudo','Lasso Lambda','Lasso Alpha Lambda','Robustfit','Ridge'},'Location','northwest','NumColumns',2)

%% Separate Integers into their own problems. 

[B, I] = sort(tr_b); %sorts training B so I can find integers using I.
[B1, I1] = sort(te_b); %sorts testing B1 so I can find integers using I1.

%% Reorganizing training A into individual A's by integer.

A = tr_A(I,:); %creates A, which has training rows ordered based on I from sort.
A1 = te_A(I1,:); %creates A1, which has testing rows ordered based on I from sort.

%% Separating A by integer value.

Zero = A(1:5923,:); %creates matrix from training data that corresponds to 0's image.
One = A(5924:12665,:); %same as above but for 1's, etc.
Two = A(12666:18623,:);
Three = A(18624:24754,:);
Four = A(24755:30596,:);
Five = A(30597:36017,:);
Six = A(36018:41935,:);
Seven = A(41936:48200,:);
Eight = A(48201:54051,:);
Nine = A(54052:60000,:);

Zero1 = A1(1:980,:); %creates matrix from testing data that corresponds to 0's image.
One1 = A1(981:2115,:); %same as above but for 1's, etc.
Two1 = A1(2116:3147,:);
Three1 = A1(3148:4157,:);
Four1 = A1(4158:5139,:);
Five1 = A1(5140:6031,:);
Six1 = A1(6032:6989,:);
Seven1 = A1(6990:8017,:);
Eight1 = A1(8018:8991,:);
Nine1 = A1(8992:10000,:); 

%% Making b values for each individual integer.

%training vectors.
b_0 = zeros(5923,1); %Outcomes for each integer, I could have done this differently most likely.
b_1 = ones(6742,1);
b_2 = 2*ones(5958,1);
b_3 = 3*ones(6131,1);
b_4 = 4*ones(5842,1);
b_5 = 5*ones(5421,1);
b_6 = 6*ones(5918,1);
b_7 = 7*ones(6265,1);
b_8 = 8*ones(5851,1);
b_9 = 9*ones(5949,1);

%test vectors
b1_0 = zeros(980,1); %Outcomes for each integer, I could have done this differently most likely.
b1_1 = ones(1135,1);
b1_2 = 2*ones(1032,1);
b1_3 = 3*ones(1010,1);
b1_4 = 4*ones(982,1);
b1_5 = 5*ones(892,1);
b1_6 = 6*ones(958,1);
b1_7 = 7*ones(1028,1);
b1_8 = 8*ones(974,1);
b1_9 = 9*ones(1009,1);

%% Zero
x0_1 = Zero\b_0; %backslash
x0_2 = pinv(Zero)*b_0; %solves Ax=b with pseudo inverse. 
x0_3 = lasso(Zero,b_0,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x0_3a = lasso(Zero,b_0,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x0_4 = robustfit(Zero,b_0); %robust fit for Ax=b.
x0_4 = x0_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x0_5 = ridge(b_0,Zero,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x0_5 = x0_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x0_1)
title('Backslash')
subplot(3,2,2)
plot(x0_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x0_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x0_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x0_4)
title('Robustfit')
subplot(3,2,6)
plot(x0_5)
title('Ridge')

%% Error

E1_x0 =  norm(b1_0-Zero1*x0_1)/norm(b1_0); %backslash
E2_x0 =  norm(b1_0-Zero1*x0_2)/norm(b1_0); %pseudo
E3_x0 =  norm(b1_0-Zero1*x0_3)/norm(b1_0); %Lasso with Lambda
E3a_x0 =  norm(b1_0-Zero1*x0_3a)/norm(b1_0); %Lasso with Lambda and Alpha
E4_x0 =  norm(b1_0-Zero1*x0_4)/norm(b1_0); %Robustfit
E5_x0 =  norm(b1_0-Zero1*x0_5)/norm(b1_0); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x0 E2_x0 E3_x0 E3a_x0 E4_x0 E5_x0];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Zero')

%% One
x1_1 = One\b_1; %backslash
x1_2 = pinv(One)*b_1; %solves Ax=b with pseudo inverse. 
x1_3 = lasso(One,b_1,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x1_3a = lasso(One,b_1,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x1_4 = robustfit(One,b_1); %robust fit for Ax=b.
x1_4 = x1_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x1_5 = ridge(b_1,One,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x1_5 = x1_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x1_1)
title('Backslash')
subplot(3,2,2)
plot(x1_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x1_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x1_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x1_4)
title('Robustfit')
subplot(3,2,6)
plot(x1_5)
title('Ridge')

%% Error

E1_x1 =  norm(b1_1-One1*x1_1)/norm(b1_1); %backslash
E2_x1 =  norm(b1_1-One1*x1_2)/norm(b1_1); %pseudo
E3_x1 =  norm(b1_1-One1*x1_3)/norm(b1_1); %Lasso with Lambda
E3a_x1 =  norm(b1_1-One1*x1_3a)/norm(b1_1); %Lasso with Lambda and Alpha
E4_x1 =  norm(b1_1-One1*x1_4)/norm(b1_1); %Robustfit
E5_x1 =  norm(b1_1-One1*x1_5)/norm(b1_1); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x1 E2_x1 E3_x1 E3a_x1 E4_x1 E5_x1];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error One')

%% Two
x2_1 = Two\b_2; %backslash
x2_2 = pinv(Two)*b_2; %solves Ax=b with pseudo inverse. 
x2_3 = lasso(Two,b_2,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x2_3a = lasso(Two,b_2,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x2_4 = robustfit(Two,b_2); %robust fit for Ax=b.
x2_4 = x2_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x2_5 = ridge(b_2,Two,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x2_5 = x2_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x2_1)
title('Backslash')
subplot(3,2,2)
plot(x2_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x2_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x2_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x2_4)
title('Robustfit')
subplot(3,2,6)
plot(x2_5)
title('Ridge')

%% Error

E1_x2 =  norm(b1_2-Two1*x2_1)/norm(b1_2); %backslash
E2_x2 =  norm(b1_2-Two1*x2_2)/norm(b1_2); %pseudo
E3_x2 =  norm(b1_2-Two1*x2_3)/norm(b1_2); %Lasso with Lambda
E3a_x2 =  norm(b1_2-Two1*x2_3a)/norm(b1_2); %Lasso with Lambda and Alpha
E4_x2 =  norm(b1_2-Two1*x2_4)/norm(b1_2); %Robustfit
E5_x2 =  norm(b1_2-Two1*x2_5)/norm(b1_2); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x2 E2_x2 E3_x2 E3a_x2 E4_x2 E5_x2];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Two')

%% Three
x3_1 = Three\b_3; %backslash
x3_2 = pinv(Three)*b_3; %solves Ax=b with pseudo inverse. 
x3_3 = lasso(Three,b_3,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x3_3a = lasso(Three,b_3,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x3_4 = robustfit(Three,b_3); %robust fit for Ax=b.
x3_4 = x3_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x3_5 = ridge(b_3,Three,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x3_5 = x3_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x3_1)
title('Backslash')
subplot(3,2,2)
plot(x3_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x3_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x3_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x3_4)
title('Robustfit')
subplot(3,2,6)
plot(x3_5)
title('Ridge')

%% Error

E1_x3 =  norm(b1_3-Three1*x3_1)/norm(b1_3); %backslash
E2_x3 =  norm(b1_3-Three1*x3_2)/norm(b1_3); %pseudo
E3_x3 =  norm(b1_3-Three1*x3_3)/norm(b1_3); %Lasso with Lambda
E3a_x3 =  norm(b1_3-Three1*x3_3a)/norm(b1_3); %Lasso with Lambda and Alpha
E4_x3 =  norm(b1_3-Three1*x3_4)/norm(b1_3); %Robustfit
E5_x3 =  norm(b1_3-Three1*x3_5)/norm(b1_3); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x3 E2_x3 E3_x3 E3a_x3 E4_x3 E5_x3];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Three')


%% Four
x4_1 = Four\b_4; %backslash
x4_2 = pinv(Four)*b_4; %solves Ax=b with pseudo inverse. 
x4_3 = lasso(Four,b_4,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x4_3a = lasso(Four,b_4,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x4_4 = robustfit(Four,b_4); %robust fit for Ax=b.
x4_4 = x4_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x4_5 = ridge(b_4,Four,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x4_5 = x4_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x4_1)
title('Backslash')
subplot(3,2,2)
plot(x4_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x4_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x4_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x4_4)
title('Robustfit')
subplot(3,2,6)
plot(x4_5)
title('Ridge')

%% Error

E1_x4 =  norm(b1_4-Four1*x4_1)/norm(b1_4); %backslash
E2_x4 =  norm(b1_4-Four1*x4_2)/norm(b1_4); %pseudo
E3_x4 =  norm(b1_4-Four1*x4_3)/norm(b1_4); %Lasso with Lambda
E3a_x4 =  norm(b1_4-Four1*x4_3a)/norm(b1_4); %Lasso with Lambda and Alpha
E4_x4 =  norm(b1_4-Four1*x4_4)/norm(b1_4); %Robustfit
E5_x4 =  norm(b1_4-Four1*x4_5)/norm(b1_4); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x4 E2_x4 E3_x4 E3a_x4 E4_x4 E5_x4];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Four')


%% Five
x5_1 = Five\b_5; %backslash
x5_2 = pinv(Five)*b_5; %solves Ax=b with pseudo inverse. 
x5_3 = lasso(Five,b_5,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x5_3a = lasso(Five,b_5,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x5_4 = robustfit(Five,b_5); %robust fit for Ax=b.
x5_4 = x5_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x5_5 = ridge(b_5,Five,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x5_5 = x5_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x5_1)
title('Backslash')
subplot(3,2,2)
plot(x5_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x5_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x5_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x5_4)
title('Robustfit')
subplot(3,2,6)
plot(x5_5)
title('Ridge')

%% Error

E1_x5 =  norm(b1_5-Five1*x5_1)/norm(b1_5); %backslash
E2_x5 =  norm(b1_5-Five1*x5_2)/norm(b1_5); %pseudo
E3_x5 =  norm(b1_5-Five1*x5_3)/norm(b1_5); %Lasso with Lambda
E3a_x5 =  norm(b1_5-Five1*x5_3a)/norm(b1_5); %Lasso with Lambda and Alpha
E4_x5 =  norm(b1_5-Five1*x5_4)/norm(b1_5); %Robustfit
E5_x5 =  norm(b1_5-Five1*x5_5)/norm(b1_5); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x5 E2_x5 E3_x5 E3a_x5 E4_x5 E5_x5];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Five')

%% Six
x6_1 = Six\b_6; %backslash
x6_2 = pinv(Six)*b_6; %solves Ax=b with pseudo inverse. 
x6_3 = lasso(Six,b_6,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x6_3a = lasso(Six,b_6,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x6_4 = robustfit(Six,b_6); %robust fit for Ax=b.
x6_4 = x6_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x6_5 = ridge(b_6,Six,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x6_5 = x6_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x6_1)
title('Backslash')
subplot(3,2,2)
plot(x6_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x6_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x6_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x6_4)
title('Robustfit')
subplot(3,2,6)
plot(x6_5)
title('Ridge')

%% Error

E1_x6 =  norm(b1_6-Six1*x6_1)/norm(b1_6); %backslash
E2_x6 =  norm(b1_6-Six1*x6_2)/norm(b1_6); %pseudo
E3_x6 =  norm(b1_6-Six1*x6_3)/norm(b1_6); %Lasso with Lambda
E3a_x6 =  norm(b1_6-Six1*x6_3a)/norm(b1_6); %Lasso with Lambda and Alpha
E4_x6 =  norm(b1_6-Six1*x6_4)/norm(b1_6); %Robustfit
E5_x6 =  norm(b1_6-Six1*x6_5)/norm(b1_6); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x6 E2_x6 E3_x6 E3a_x6 E4_x6 E5_x6];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Six')

%% Seven
x7_1 = Seven\b_7; %backslash
x7_2 = pinv(Seven)*b_7; %solves Ax=b with pseudo inverse. 
x7_3 = lasso(Seven,b_7,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x7_3a = lasso(Seven,b_7,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x7_4 = robustfit(Seven,b_7); %robust fit for Ax=b.
x7_4 = x7_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x7_5 = ridge(b_7,Seven,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x7_5 = x7_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x7_1)
title('Backslash')
subplot(3,2,2)
plot(x7_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x7_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x7_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x7_4)
title('Robustfit')
subplot(3,2,6)
plot(x7_5)
title('Ridge')

%% Error

E1_x7 =  norm(b1_7-Seven1*x7_1)/norm(b1_7); %backslash
E2_x7 =  norm(b1_7-Seven1*x7_2)/norm(b1_7); %pseudo
E3_x7 =  norm(b1_7-Seven1*x7_3)/norm(b1_7); %Lasso with Lambda
E3a_x7 =  norm(b1_7-Seven1*x7_3a)/norm(b1_7); %Lasso with Lambda and Alpha
E4_x7 =  norm(b1_7-Seven1*x7_4)/norm(b1_7); %Robustfit
E5_x7 =  norm(b1_7-Seven1*x7_5)/norm(b1_7); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x7 E2_x7 E3_x7 E3a_x7 E4_x7 E5_x7];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Seven')

%% Eight
x8_1 = Eight\b_8; %backslash
x8_2 = pinv(Eight)*b_8; %solves Ax=b with pseudo inverse. 
x8_3 = lasso(Eight,b_8,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x8_3a = lasso(Eight,b_8,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x8_4 = robustfit(Eight,b_8); %robust fit for Ax=b.
x8_4 = x8_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x8_5 = ridge(b_8,Eight,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x8_5 = x8_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x8_1)
title('Backslash')
subplot(3,2,2)
plot(x8_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x8_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x8_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x8_4)
title('Robustfit')
subplot(3,2,6)
plot(x8_5)
title('Ridge')

%% Error

E1_x8 =  norm(b1_8-Eight1*x8_1)/norm(b1_8); %backslash
E2_x8 =  norm(b1_8-Eight1*x8_2)/norm(b1_8); %pseudo
E3_x8 =  norm(b1_8-Eight1*x8_3)/norm(b1_8); %Lasso with Lambda
E3a_x8 =  norm(b1_8-Eight1*x8_3a)/norm(b1_8); %Lasso with Lambda and Alpha
E4_x8 =  norm(b1_8-Eight1*x8_4)/norm(b1_8); %Robustfit
E5_x8 =  norm(b1_8-Eight1*x8_5)/norm(b1_8); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x8 E2_x8 E3_x8 E3a_x8 E4_x8 E5_x8];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Eight')


%% Nine
x9_1 = Nine\b_9; %backslash
x9_2 = pinv(Nine)*b_9; %solves Ax=b with pseudo inverse. 
x9_3 = lasso(Nine,b_9,'Lambda',0.5); %solves Ax=b with Lasso. 1-norm penalty.
x9_3a = lasso(Nine,b_9,'Lambda',0.5,'Alpha',0.5); %uses 1-norm and 2-norm penalty.
x9_4 = robustfit(Nine,b_9); %robust fit for Ax=b.
x9_4 = x9_4(1:784); %rank deficient and 785x1. Not sure how to fix.
x9_5 = ridge(b_9,Nine,0.01,0); %ridge for Ax=b. I used 0.01 for k.
x9_5 = x9_5(1:784); %rank deficient and 785x1.

% Plots for Training Data

subplot(3,2,1)
plot(x9_1)
title('Backslash')
subplot(3,2,2)
plot(x9_2)
title('Pseduo Inverse')
subplot(3,2,3)
plot(x9_3)
title('Lasso Lambda')
subplot(3,2,4)
plot(x9_3a) 
title('Lasso Lambda Alpha')
subplot(3,2,5)
plot(x9_4)
title('Robustfit')
subplot(3,2,6)
plot(x9_5)
title('Ridge')

%% Error

E1_x9 =  norm(b1_9-Nine1*x9_1)/norm(b1_9); %backslash
E2_x9 =  norm(b1_9-Nine1*x9_2)/norm(b1_9); %pseudo
E3_x9 =  norm(b1_9-Nine1*x9_3)/norm(b1_9); %Lasso with Lambda
E3a_x9 =  norm(b1_9-Nine1*x9_3a)/norm(b1_9); %Lasso with Lambda and Alpha
E4_x9 =  norm(b1_9-Nine1*x9_4)/norm(b1_9); %Robustfit
E5_x9 =  norm(b1_9-Nine1*x9_5)/norm(b1_9); %Ridge

%uses error from lecture norm(test_b - test_A*train_x)/norm(test_b).
Error = [E1_x9 E2_x9 E3_x9 E3a_x9 E4_x9 E5_x9];
X = categorical({'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
X = reordercats(X,{'Backslash','Pseudo','Lasso Lambda','Lasso Lambda Alpha','Robustfit','Ridge'});
bar(X,Error)
title('Error Nine')

%%
subplot(3,4,1)
pcolor(reshape(x2_tr,28,28))
title('Heat Map All')
subplot(3,4,2)
pcolor(reshape(x0_1,28,28))
title('Heat Map Zero')
subplot(3,4,3)
pcolor(reshape(x1_3,28,28))
title('Heat Map One')
subplot(3,4,4)
pcolor(reshape(x2_2,28,28))
title('Heat Map Two')
subplot(3,4,5)
pcolor(reshape(x3_2,28,28))
title('Heat Map Three')
subplot(3,4,6)
pcolor(reshape(x4_3,28,28))
title('Heat Map Four')
subplot(3,4,7)
pcolor(reshape(x5_3,28,28))
title('Heat Map Five')
subplot(3,4,8)
pcolor(reshape(x6_2,28,28))
title('Heat Map Six')
subplot(3,4,9)
pcolor(reshape(x7_2,28,28))
title('Heat Map Seven')
subplot(3,4,10)
pcolor(reshape(x8_2,28,28))
title('Heat Map Eight')
subplot(3,4,11)
pcolor(reshape(x9_2,28,28))
title('Heat Map Nine')
