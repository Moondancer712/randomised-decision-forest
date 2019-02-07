function [dataTrain, dataQuery] = codebook_kmeans(nClusters, nDescriptors, nSamples, folderName, classList, showImg, descType)
% Function:
%   - generate visual codebook by k-means method
%
% InputArg(s):
%   - nClusters: number of clusters (size of codebook)
%   - nDescriptors: number of descriptors for clustering
%   - nSamples: number of samples for train and test per class without
%  replacement (assume equal)
%   - folderName: image directory
%   - classList: category name
%   - showImg: show image or not
%   - descType: criteria for obtaining descriptors
%
% OutputArg(s):
%   - dataTrain: vectorised training data with label
%   - dataQuery: vectorised test data without label
%
% Restraints:
%   - the split function only supports information gain so far
%
% Comments:
%   - the RF Codebook is supposed to give better results than the K-means one
%
% Author & Date: Yang (i@snowztail.com) - 05 Feb 19

% number of classes
nClasses = length(classList);
%% Training data: feature detection and descriptors extraction
dataType = 'train';
disp('Loading training images...');
tic;
[descTrain, imgIdxTrain] = feature_detection(classList, folderName, nSamples, dataType, descType);
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
%% K-means clustering
disp('Clustering data by K-means...');
tic;
% compute the centroids position
[centroid, ~] = vl_kmeans(descSel, nClusters);
centroid = centroid';
toc;
%% Training data: assign patch descriptors to the visual codebook (vector quantisation)
dataType = 'train';
disp('Encoding training images...');
tic;
[dataTrain] = vector_quantisation_knn(classList, folderName, nSamples, nClusters, imgIdxTrain, centroid, descTrain, dataType, showImg);
toc;
%% Testing data: feature detection and descriptors extraction
dataType = 'test';
disp('Loading testing images...');
tic;
[descTest, imgIdxTest] = feature_detection(classList, folderName, nSamples, dataType, descType);
toc;
%% Testing data: assign patch descriptors to the visual codebook (vector quantisation)
dataType = 'test';
disp('Encoding testing images...');
[dataQuery] = vector_quantisation_knn(classList, folderName, nSamples, nClusters, imgIdxTest, centroid, descTest, dataType, showImg);
end

