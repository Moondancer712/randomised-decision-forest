clear; close all;
%% the parameters of RF (for each tree)
% number of candidate weak learners 
rf.splitNum = 10;
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
classList = {classList(3:end).name};
% number of image classes
nClasses = length(classList);
% multi-resolution (values determine the scale of each layer)
phowSize = [4 8 10];
% step size (the lower the denser, select from {2, 4, 8, 16})
phowStep = 8;
%% Obtain codebook by K-means
[dataTrain, dataQuery] = codebook_kmeans(nClusters, nDescriptors, nSamples, folderName, classList, phowSize, phowStep, showImg);
%% Build random forest by training data and predetermined parameters
forest = growTrees(dataTrain, rf);
%% Classify the training data by random forest
[accuTrain, confTrain] = classification(nClasses, dataTrain, forest, showHist);
%% Classify the testing data by random forest
[accuTest, confTest] = classification(nClasses, dataQuery, forest, showHist);
