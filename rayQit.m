function [eigval, eigvec,iter] = rayQit(A,v)
%Uses Rayleigh Quotient iteration to find the largest eigenvalues and 
%eigenvectors of a matrix based on guesses in matrix A.
[~,n]=size(A);%gives row and column size of A.
iter = 0; %keeps track of iteration number.
eigvec = v; %normalizes guess.
eigval = eigvec'*A*eigvec; %guess for first eigenvalue.
for i = 1:12 %loop to converge on eigenvalue and vector of first guess.
    iter = iter+1;
    w = (A - eigval*eye(n,n))\eigvec; %iteration for eigenvector.
    eigvec = w/norm(w); %normalizes iteration eigenvector.
    eigval = eigvec'*A*eigvec; %calculates corresponding eigenvalue.
    if norm((A - eigval*eye(n,n))*eigvec) < 10^-8 %checks tolerance.
        break
    end
end


end

