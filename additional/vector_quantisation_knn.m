function [data] = vector_quantisation_knn(classList, folderName, nSamples, nClusters, imgIdxSet, centroid, descriptor, type, showImg)
% Function:
%   - assign patch descriptors to the visual codebook by k-nearest
%   neighbors algorithm
%
% InputArg(s):
%   - classList: the classes (labels) of the image set
%   - folderName: the directory of the image set
%   - nSamples: number of images per class
%   - nClusters: number of clusters or dimensions in KNN method
%   - imgIdxSet: the image index set
%   - centroid: the centers of descriptors of different categories
%   - descriptor: base elements to describe an image
%   - type: data type (train or test)
%   - showImg: whether to show images and histograms
%
% OutputArg(s):
%   - data: the frequency of codewords for histogram with or without label
%
% Comments:
%   - link descriptors to codewords
%
% Author & Date: Yang (i@snowztail.com) - 04 Feb 19

% number of classes
nClasses = length(classList);
% frequency of descriptors for training or testing dataset (for histogram)
frequency = zeros(nClasses * nSamples, nClusters);
% figures label
label = zeros(nClasses * nSamples, 1);
for iClass = 1: nClasses
    % get image directory
    subFolderName = fullfile(folderName, classList{iClass});
    imgList = dir(fullfile(subFolderName, '*.jpg'));
    if showImg
        figure;
        suptitle('Training image representations: 256-D histograms');
    end
    for iSample = 1: nSamples
        % update descriptors in current image
        descCurr = single(descriptor{iClass, iSample});
        % categorise by KNN based on obtained centroids
        index = knnsearch(centroid, descCurr');
        % compute frequency of clusters for histogram
        frequency((iClass - 1) * nSamples + iSample, :) = histcounts(index, nClusters) / length(index);
        % display training images and corresponding histograms
        if showImg
            img = imread(fullfile(subFolderName, imgList(imgIdxSet(iClass, iSample)).name));
            subplot(ceil(nSamples / 3), 6 ,2 * iSample - 1);
            imshow(img);
            subplot(ceil(nSamples / 3), 6 ,2 * iSample);
            histogram(index, nClusters);
            xlim([0 nClusters + 1]);
            drawnow;
        end
    end
    % update corresponding labels
    switch type
        case 'train'
            % label training data; leave blank for testing data
            label((iClass - 1) * nSamples + 1: iClass * nSamples) = ones(nSamples, 1) * iClass;
    end
end
% label the data
data = [frequency, label];
end

