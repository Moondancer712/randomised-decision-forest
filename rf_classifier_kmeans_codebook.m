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
showHist = 1;
% showImg: show image or not
showImg = 0;
% number of clusters (size of codebook)
nClusters = 256;
%% Obtain codebook by K-means
[data_train, data_query] = getData('Caltech', nClusters, showImg);
