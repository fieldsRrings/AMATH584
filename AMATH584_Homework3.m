clc;
clf;
close all;
clear variables;

%% Problem 1

A1 = rand(10,10); %random matrix A.
[Q,R] = GramSchmidt(A1); %call to my modified Gram-Schmidt algorithm.
B = Q'*Q; %testing my algorith. It returns diagonal of all 1's but the 
          %rest is not zero so not sure.
[M,N] = qrfactor(A1); %call to QR function made in lecture.
[X,Y] = qr(A1); %call to Matlab QR function.
cond(A1) %condition number of A. It was much greater than 1. As expected.

subplot(3,1,1) %plots my figures for the three different methods.
bar(Q(:,1))    %I noticed that the algorithm I produced makes similar 
               %vectors in magnitude but not direction.

subplot(3,1,2)
bar(M(:,1))

subplot(3,1,3)
bar(X(:,1))

%% Problem 2
x = [1.920:0.001:2.080]; %x span for our functions.
%two functions, one with expanded form, one with compact form. 
f =@(x) x.^9-18*x.^8+144*x.^7-672*x.^6+2016*x.^5-4032*x.^4+5376*x.^3-4608*x.^2+2304*x-512;
g =@(x) (x-2).^9;

y = f(x); %call to functions.
z = g(x);

%plots the functions. g(x) is much nicer than f(x), which I think is the
%point given truncation errors.
plot(x,y)
hold on
plot(x,z)
legend('Expanded','Compact') %label so you know what you're looking at.

%% Problem 3

% part a
vec_cond1 = zeros(1,20); %vector to store condition numbers.

%for loop to generate matrices of increasing size and then storing their
%condition number.
for i=1:20
    A3 = randn(12+i,10+i);
    vec_cond1(1,i) = cond(A3);
end

%plot to look at how the condition number changes as A increases. I think
%I could have done this differently. I was trying to keep the same matrix A
%and just append columns and rows to it but Matlab didn't like that. The 
%condition number stayed well above zero the entire time.
plot(vec_cond1)

% part b

A4 = randn(11,10); %new random matrix for part b.
vec2 = A4(:,1); %pulled first column and stored in vec2.
B4 = [A4 vec2]; %appended column to the end of A4.
cond(A4) %condition number of A4 without appneded column.
cond(B4) %condition number with appneded column. It was much larger.
det(B4) %the Determinant was also huge.

%part c

vec_cond2 = zeros(1,20); %vector to add noise to.
noise = rand(11,1); %vector with noise.
 %loop to carry out iterations adding noise to the vectors.
 %then storing the condition number to see how it changes.
for j=1:20
    vec3 = vec2+ j^2*noise;
    C4 = [A4 vec3];
    vec_cond2(1,j) = cond(C4);
end
%plot of condition number.
plot(vec_cond2);