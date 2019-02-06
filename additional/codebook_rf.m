function [dataTrain, dataQuery] = codebook_rf(rf, nDescriptors, nSamples, folderName, classList, phowSize, phowStep, showImg)
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
type = 'train';
disp('Loading training images...');
[descTrain, imgIdxTrain] = feature_detection(classList, folderName, nSamples, phowSize, phowStep, type);
%% Build visual vocabulary (codebook) for 'Bag-of-Words' method
disp('Building visual codebook...')
descSel = cell(nClasses, 1);
% randomly select SIFT descriptors for clustering
for iClass = 1: nClasses
    % choose same amount of descriptors per class
    descSel{iClass} = single(vl_colsubset(cat(2, descTrain{iClass, :}), nDescriptors / nClasses));
end
% combine subsets
descSel = cat(2, descSel{:});
%% Grow the forest
% develop RF based on selected descriptors and predetermined parameters
forest = growTrees(descSel', rf);
% number of clusters
nClusters = size(forest(1).prob,1);
%% Training data: assign patch descriptors to the visual codebook (vector quantisation)
disp('Encoding training images...');
type = 'train';
[dataTrain] = vector_quantisation_rf(classList, folderName, nSamples, nClusters, imgIdxTrain, forest, descTrain, type, showImg);
%% Testing data: feature detection and descriptors extraction
type = 'test';
disp('Loading testing images...');
[descTest, imgIdxTest] = feature_detection(classList, folderName, nSamples, phowSize, phowStep, type);
%% Testing data: assign patch descriptors to the visual codebook (vector quantisation)
disp('Encoding testing images...');
type = 'test';
[dataQuery] = vector_quantisation_rf(classList, folderName, nSamples, nClusters, imgIdxTest, forest, descTest, type, showImg);
end
