function [descriptor, imgIdxSet] = feature_detection(classList, folderName, nSamples, phowSize, phowStep, dataType)
% Function:
%   - detect class features and extract numerous desctriptors
%
% InputArg(s):
%   - classList: the classes (labels) of the image set
%   - folderName: the directory of the image set
%   - nSamples: number of images per class
%   - phowSize: determine the scale of each layer
%   - phowStep: the lower the denser, select from {2,4,8,16}
%   - dataType: data type (train or test)
%
% OutputArg(s):
%   - descriptor: base elements to describe an image
%   - imgIdxSet: the image index set
%
% Comments:
%   - extract PHOW features (multi-scaled dense SIFT)
%
% Author & Date: Yang (i@snowztail.com) - 04 Feb 19

% number of classes
nClasses = length(classList);
% initialisation
imgIdx = cell(nClasses, 1);
descriptor = cell(nClasses, nSamples);
imgIdxSet = zeros(nClasses, nSamples);
for iClass = 1: nClasses
    subFolderName = fullfile(folderName, classList{iClass});
    % class directory
    imgList = dir(fullfile(subFolderName, '*.jpg'));
    % randomly choose images from the class
    imgIdx{iClass} = randperm(length(imgList));
    % obtain image index (first nSamples for training, next nSamples for testing)
    switch dataType
        case 'train'
            imgIdxSet(iClass, :) = imgIdx{iClass}(1: nSamples);
        otherwise
            imgIdxSet(iClass, :) = imgIdx{iClass}(nSamples + 1: 2 * nSamples);
    end
    for iSample = 1: nSamples
        % read image
        img = imread(fullfile(subFolderName, imgList(imgIdxSet(iClass, iSample)).name));
        % if the image is not in gray scale
        if size(img, 3) == 3
            % PHOW work on gray scale image
            img = rgb2gray(img);
        end
        % obtain training or testing descriptors
        [~, descriptor{iClass, iSample}] = vl_phow(single(img), 'Sizes', phowSize, 'Step', phowStep);
    end
end
end

