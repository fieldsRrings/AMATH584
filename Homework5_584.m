clc;
close all;

m = 10; %dimension for matrix.
A = rand(m,m); %random m by m matrix.

%% Symmetric Matrix.

B = A*A'; %symmetric matrix.
[Ve1,D1] = eigs(B,10); %returns m eigenvectors and eigenvalues for B.
[val1,vec1,iter1]= powit(B); %call to powit.

rayval_1 = zeros(m,1); %Column vector for eigenvalues from rayQit.
rayvec_1 = zeros(m,m); %Matrix for eigenvectors from rayQit.
iter_ray1 = zeros(m,1); %Column vector to keep track of iterations.
for k=1:10
    v1 = Ve1(:,k)*2 + 0.1; %"guess" for rayQit based on modified eigenvectors.
    [rayval_1(k,1),rayvec_1(:,k),iter_ray1(k,1)] = rayQit(B,v1);
end

%% Non Symmetric Matrix.

[Ve2,D2] = eigs(A,10); %returns m eigenvectors and eigenvalues from A.
[val3,vec3,iter2]= powit(A); %call to power iteration function.

rayval_2 = zeros(m,1); %same as above.
rayvec_2 = zeros(m,m);
iter_ray2 = zeros(m,1);

for i=1:10
    v2 = Ve2(:,i)*2 + 0.01;
    [rayval_2(i,1),rayvec_2(:,i),iter_ray2(i,1)] = rayQit(A,v2);   
end

%% Plots of symmetric and non symmetric.

figure
subplot(2,1,1)
plot(real(diag(D1)),imag(diag(D1)),'rs',real(rayval_1),imag(rayval_1),'kd')
xlabel('Real')
ylabel('Imaginary')
title('Symmetric')
legend({'Eigs','Rayleigh'})

subplot(2,1,2)
plot(real(diag(D2)),imag(diag(D2)),'g*',real(rayval_2),imag(rayval_2),'bo')
xlabel('Real')
ylabel('Imaginary')
title('Non Symmetric')
legend({'Eigs','Rayleigh'})

%% Download Yale faces.
 
% Specify the folder where the files live.
myFolder = 'C:\Users\Richard\Downloads\yalefaces_cropped\CroppedYale';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder2);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); % Ask for a new one.
    if myFolder == 0
         % User clicked Cancel
         return;
    end
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '**/*.pgm'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
C = zeros(9600, length(theFiles)); %Creates matrix to store images.
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    imageArray = imresize(double(imread(fullFileName)),[120 80]);
    %Converts int to double.
    imageVector = imageArray(:);
    %Converts columns of imageArray into single column vector.
    C(:,k) = imageVector;
    %Stores imageVector as column vector of A.
end

%% SVD and Randomized Linear Algebra of Yale Faces.

R = corrcoef(C); %Square correlation coefficient matrix.
[yale_val,yale_vec,yale_it] = powit(R); %call to powit for yalefaces.
[U,S,V] = svd(C); %SVD of yalefaces.
[U1,S1,V1] = ran_lin(C); %randomized linear algebra SVD.

%% Error

S = S(1:1207,1:1207); %submatrix of singular values from SVD.
errs = diag(S); %singular values from SVD.
errs1 = diag(S1); %singular values from randomized SVD.
error = abs(mean(errs-errs1)); %error between the two. 