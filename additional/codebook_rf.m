function [dataTrain, dataQuery] = codebook_rf(rf, nDescriptors, nSamples, folderName, classList, phowSize, phowStep, showImg, wlType)
% Function:
%   - generate visual codebook by randomised decision forest
%
% InputArg(s):
%   - rf: the parameters of RF (for each tree)
%       - rf.splitNum: number of candidate weak learners
%       - rf.depth: number of layers
%       - rf.split: criteria in split decision (information gain, gain ratio, etc)
%       - rf.num: number of trees
%   - nDescriptors: number of descriptors for clustering
%   - nSamples: number of samples for train and test per class without
%  replacement (assume equal)
%   - folderName: image directory
%   - classList: category name
%   - phowSize: values determine the scale of each layer
%   - phowStep: step size (the lower the denser)
%   - showImg: show image or not
%   - wlType: type of the weak learner (now support 'axis-aligned' and '2-pixel' test)
%
% OutputArg(s):
%   - dataTrain: vectorised training data with label
%   - dataQuery: vectorised test data without label
%   - nClasses: number of image classes
%
% Restraints:
%   - the split function only supports information gain so far
%
% Comments:
%   - the RF Codebook is supposed to give better results than the K-means one
%
% Author & Date: Yang (i@snowztail.com) - 02 Feb 19

% number of classes
nClasses = length(classList);
%% Training data: feature detection and descriptors extraction
dataType = 'train';
disp('Loading training images...');
tic;
[descTrain, imgIdxTrain] = feature_detection(classList, folderName, nSamples, phowSize, phowStep, dataType);
toc;
%% Build visual vocabulary (codebook) for 'Bag-of-Words' method
disp('Building visual codebook...');
tic;
descSel = cell(nClasses, 1);
% randomly select SIFT descriptors for clustering
for iClass = 1: nClasses
    % choose same amount of descriptors per class
    descSel{iClass} = single(vl_colsubset(cat(2, descTrain{iClass, :}), nDescriptors / nClasses));
end
% combine subsets
descSel = cat(2, descSel{:});
toc;
%% Grow the forest
disp('Clustering data by random forest...');
tic;
% develop RF based on selected descriptors and predetermined parameters
forest = growTrees(descSel', rf, wlType);
% number of clusters
nClusters = size(forest(1).prob,1);
toc;
%% Training data: assign patch descriptors to the visual codebook (vector quantisation)
dataType = 'train';
disp('Encoding training images...');
tic;
[dataTrain] = vector_quantisation_rf(classList, folderName, nSamples, nClusters, imgIdxTrain, forest, descTrain, dataType, showImg, wlType);
toc;
%% Testing data: feature detection and descriptors extraction
dataType = 'test';
disp('Loading testing images...');
tic;
[descTest, imgIdxTest] = feature_detection(classList, folderName, nSamples, phowSize, phowStep, dataType);
toc;
%% Testing data: assign patch descriptors to the visual codebook (vector quantisation)
dataType = 'test';
disp('Encoding testing images...');
tic;
[dataQuery] = vector_quantisation_rf(classList, folderName, nSamples, nClusters, imgIdxTest, forest, descTest, dataType, showImg, wlType);
toc;
end

