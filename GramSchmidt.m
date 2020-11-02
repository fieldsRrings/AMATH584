function [Q,R] = GramSchmidt(A)
%Modified Gram Schmidt from Trefethen.
%   Breaks A into unitary Q and upper triangular R matrices.
%size of A
[m,n] = size(A);
%creates new matrices Q and R with zero entries.
Q = zeros(m,n);
R = zeros(n,n);
%loop to carry out modified Gram-Schmidt from Trefethen.
for i=1:n
    R(i,i) = norm(A(:,i)); %norm of a column vector from A stored into R.
    Q(:,i) = A(:,i)./R(i,i); %normalized column vector from A stored in Q.
    for j=(i+1):n
        R(i,j) = Q(:,i)'*A(:,j); %fills the column vector above the diagonal of R.
        Q(:,j) = Q(:,j) - R(i,j)*Q(:,i); %orthogonal vector.
    end
end
end

