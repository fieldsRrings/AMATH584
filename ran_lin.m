function [U,S,V] = ran_lin(A)
%SVD using Randomized Linear Algebra
%ran_lin takes A and multiplies it by random matrix omega to get random
%sample of A. Then performs QR Decomposition on new matrix to get SVD.
[~,n]=size(A); %Size of A.
W = randn(n,n/2); %Random matrix
Y = A*W; %Random sample of A.
[Q,~]= qr(Y); %QR Decomposition of A.
B = Q'*A; %Embeds A into columns of Q'.
[U,S,V]=svd(B); %SVD of B to get feature space.
S = S(1:n/2,1:n/2); %Singular Values. 
U = U(1:n/2,1:n/2); %Feature Space.
V = V(1:n/2,1:n/2); %Right side vectors.
end

