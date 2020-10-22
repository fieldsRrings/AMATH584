%%

%This code was linked on Piazza, I modified it a little.
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
A = zeros(9600, length(theFiles)); %Creates matrix to store images.
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    imageArray = imresize(double(imread(fullFileName)),[120 80]);
    %Converts int to double.
    imageVector = imageArray(:);
    %Converts columns of imageArray into single column vector.
    A(:,k) = imageVector;
    %Stores imageVector as column vector of A.
end

%%

%Matrix to store uncropped column vector images.
B = zeros(9600, 165);

%List of files within the uncropped.tar Yale folder.
uncroppedFiles = dir('C:\Users\Richard\Downloads\yalefaces_uncropped\yalefaces');

%Loop to iterate through the list of uncroppeed faces and store them as
%vectors.
for m = 1: 165
    %Finds an uncropped image, converts it to double, resizes it and stores it as an
    %array.
    uncroppedArray = imresize(double(imread(strcat('C:\Users\Richard\Downloads\yalefaces_uncropped\yalefaces\',(uncroppedFiles(m+2).name)))), [120 80]);
    %Converts the array image into a column vector.
    uncroppedVector = uncroppedArray(:);
    %Stores the uncropped column vector as a column vector in B.
    B(:,m)= uncroppedVector;
end
%%

%SVD on A
[U1,S1,V1] = svd(A,'econ');
%Removes the Singular Values from S1.
x1 = diag(S1);
%Gives the Rank of A, which should match the singular values of S1. 
r1 = rank(A);

%%

%SVD on B
[U2,S2,V2] = svd(B,'econ');
%Removes the Singular Values from S2.
x2 = diag(S2);
%Gives the Rank of B, which should match the singular values of S2. 
r2 = rank(B);

%%

%Plots the first few columns of matrix U1 from the SVD A.
figure;
hold on
for i = 1:3
 plot(U1(:,i))
end

%%

%Plots the first few columns of matrix U2 from the SVD B.
figure;
hold on
for i = 1:3
 plot(U2(:,i))
end

%%
figure;
%Plots the first modes of PCA required to differentiate images.
semilogy(x1(1:400))

%%
figure;
%Plots the first modes of PCA required to differentiate images.
semilogy(x2(1:165))

%%

figure(1);
%Original image from column vector of A.
image(reshape(A(:,10), 120, 80))

%%
%New matrix with selected mode value. 
A1 = U1(:,1:300)*S1(1:300,1:300)*V1(:,1:300).';

figure(2);
%New image from selected mode value.
image(reshape(A1(:,10), 120, 80))

%%

figure(1);
%Original image from column vector of B.
image(reshape(B(:,10), 120, 80))

%%
%New matrix with selected mode value. 
B1 = U2(:,1:165)*S2(1:165,1:165)*V2(:,1:165).';

figure(2);
%New image from selected mode value.
image(reshape(B1(:,10), 120, 80))

