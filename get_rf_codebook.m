clear; close all; clc;
% cd ('..');
% init;
% number of weak learners
param.num = 10;
% trees depth
param.depth = 10;
% number of candidate weak learners
param.splitNum = 3;
% criteria in split decision
param.split = 'IG';
% whether to show image
showImg = false;
[data_train, data_query] = rf_codebook(param, showImg);
