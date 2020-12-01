function [U,S,V,approx] = ran_lin(A,k)
%SVD using Randomized Linear Algebra
%ran_lin takes A and multiplies it by random matrix omega to get random
%sample of A. Then performs QR Decomposition on new matrix to get SVD.
[~,n]=size(A); %Size of A.
W = randn(n,k); %Random matrix
Y = A*W; %Random sample of A.
[Q,~]= qr(Y,0); %QR Decomposition of A.
B = Q'*A; %Embeds A into columns of Q'.
[U,S,V]=svd(B,'econ'); %SVD of B to get feature space.
approx = Q*U; %Back to U matrix for svd(A).
end

