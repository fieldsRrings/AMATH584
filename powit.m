function [eigval, eigvec, iter] = powit(A)
%Power Iteration of A.
%Calculates largest eigenvalue and corresponding eigenvector of matrix A
%using power iteration.
iter = 0; %keeps track of iteration number.
[~,n]=size(A); %row and column dimensions of A.
v=rand(n,1); %initial random guess for solution.
eigvec=v/norm(v); %normalize random eigenvector guess for iteration.
error = 1; %initial error conditions to force into while loop.
err2 = 10; %initial error guess. Not sure if this number matters. 
while error > 10^-10
    iter = iter+1;
    w = A*eigvec; %applies A to eigenvector.
    err1 = max(abs(w)); %pulls out largest magnitude value for error.
    eigvec = w/norm(w); %normalizes eigenvector.
    eigval = eigvec'*A*eigvec; %calculates eigenvalue using new eigenvector.
    error = abs(err2 - err1); %error
    err2 = err1; %changes err2 value for next iteration.
end
end

