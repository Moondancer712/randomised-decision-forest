clear; close all; clc; ticRf = tic;
%% Parameters of RF (for each tree)
% number of candidate weak learners 
rf.splitNum = 5;
% number of layers
rf.depth = 5;
% criteria in split decision (information gain)
rf.split = 'IG';
% number of trees
rf.num = 10;
%% Initialisation
% show decision histogram or not
showHist = false;
% whether to show image
showImg = false;
% whether to show confusion matrix
showConf = true;
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
% criteria for obtaining descriptors
% SIFT is to detect and describe local features in images
% descType.name = 'sift';
% extracts a dense set of SIFT features from image
% descType.name = 'dsift';
% detects upright scale and translation covariant features based on the Difference of Gaussian (DoG) cornerness
% descType.name = 'covdet';
% phow is simply dense SIFT applied at several resolutions
descType.name = 'phow';
% multi-resolution (values determine the scale of each layer)
descType.size = [4 8 10];
% step size (the lower the denser, select from {2, 4, 8, 16})
descType.step = 8;
% weak learner type
% wlType = 'axis-aligned';
% wlType = '2-pixel';
wlType = 'linear';
%% Obtain codebook by random forest
disp('Obtaining codebook by random forest...');
disp('--------------------------------------------------');
tic;
[dataTrain, dataQuery] = codebook_rf(rf, nDescriptors, nSamples, folderName, classList, showImg, wlType, descType);
disp('--------------------------------------------------');
toc;
%% Build random forest by training data and predetermined parameters
disp('==================================================');
disp('Building random forest...');
tic;
forest = growTrees(dataTrain, rf, wlType);
toc;
%% Classify the training data by random forest
disp('==================================================');
disp('Classifying training data...');
tic;
[accuTrain, confTrain] = classification(nClasses, dataTrain, forest, showHist, showConf, wlType);
toc;
fprintf('The accuracy for training data is %.2f %%.\n', 100 * accuTrain);
%% Classify the testing data by random forest
disp('==================================================');
disp('Classifying testing data...');
tic;
[accuTest, confTest] = classification(nClasses, dataQuery, forest, showHist, showConf, wlType);
toc;
fprintf('The accuracy for testing data is %.2f %%.\n', 100 * accuTest);
%% Elapsed time
disp('==================================================');
tocRf = toc(ticRf);
fprintf('The overall time cost is %f seconds.\n', tocRf);
