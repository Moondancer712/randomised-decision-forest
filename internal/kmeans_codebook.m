clear; close all; clc;
cd ('..');
init;
%% Train and test
nClusters = 256;
% showImg = true;
showImg = false;
[data_train, data_test] = getData('Caltech', nClusters, showImg);
