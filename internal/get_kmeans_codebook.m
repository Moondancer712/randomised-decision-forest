clear; close all; clc;
% cd ('..');
% init;
%% Train and test
nClusters = 256;
% Show training & testing images and their image feature vector (histogram representation)
showImg = false;
[data_train, data_test] = getData('Caltech', nClusters, showImg);
