clear; close all; clc; ticKmeans = tic;
%% Parameters of RF (for each tree)
% number of candidate weak learners 
rf.splitNum = 5;
% number of layers
rf.depth = 5;
% criteria in split decision (information gain)
rf.split = 'IG';
% number of trees
rf.num = 50;
%% Initialisation
% show decision histogram or not
showHist = false;
% whether to show image
showImg = false;
% whether to show confusion matrix
showConf = true;
% number of clusters (size of codebook)
nClusters = 256;
% size of descriptors for clustering
nDescriptors = 1e4;
% number of samples for train and test per class without
% replacement (assume equal)
nSamples = 15;
% image directory
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
% choose classes
classList = {classList(3: end).name};
% number of image classes
nClasses = length(classList);
% multi-resolution (values determine the scale of each layer)
phowSize = [4 8 10];
% step size (the lower the denser, select from {2, 4, 8, 16})
phowStep = 8;
%% Obtain codebook by K-means
disp('Obtaining codebook by K-means...');
disp('--------------------------------------------------');
tic;
[dataTrain, dataQuery] = codebook_kmeans(nClusters, nDescriptors, nSamples, folderName, classList, phowSize, phowStep, showImg);
disp('--------------------------------------------------');
toc;
%% Build random forest by training data and predetermined parameters
disp('==================================================');
disp('Building random forest...');
tic;
forest = growTrees(dataTrain, rf);
toc;
%% Classify the training data by random forest
disp('==================================================');
disp('Classifying training data...');
tic;
[accuTrain, confTrain] = classification(nClasses, dataTrain, forest, showHist, showConf);
toc;
fprintf('The accuracy for training data is %.2f %%.\n', 100 * accuTrain);
%% Classify the testing data by random forest
disp('==================================================');
disp('Classifying testing data...');
tic;
[accuTest, confTest] = classification(nClasses, dataQuery, forest, showHist, showConf);
toc;
fprintf('The accuracy for testing data is %.2f %%.\n', 100 * accuTest);
%% Elapsed time
disp('==================================================');
tocKmeans = toc(ticKmeans);
fprintf('The overall time cost is %f seconds.\n', tocKmeans);
