function [data_train, data_query] = rf_codebook(param, showImg)
% Function:
%   - generate visual codebook by randomised decision forest
%
% InputArg(s):
%   - param: the parameters of RF (for each tree)
%       - param.splitNum: number of candidate weak learners 
%       - param.depth: number of layers
%       - param.split: criteria in split decision (information gain, gain ratio, etc)
%       - param.num: number of trees
%   - showImg: show image or not
%
% OutputArg(s):
%   - data_train: vectorised training data with label
%   - data_query: vectorised test data without label
%
% Restraints:
%   - the split function only supports information gain so far
%
% Comments:
%   - the RF Codebook is supposed to give better results than the K-means one
%
% Author & Date: Yang (i@snowztail.com) - 02 Feb 19

%% Initialisation
% Multi-resolution, these values determine the scale of each layer
PHOW_Sizes = [4 8 10];
% The lower the denser. Select from {2,4,8,16}
PHOW_Step = 16;
% randomly select 15 images each class without replacement for training and testing
imgSel = [15 15];
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
% choose classes
classList = {classList(3:end).name};
% number of classes
nClasses = length(classList);
% number of samples for train and test per class
nSamplesTrain = imgSel(1);
nSamplesTest = imgSel(2);
% initialisation
imgIdx = cell(nClasses, 1);
descTrain = cell(nClasses, nSamplesTrain);
descTest = cell(nClasses, nSamplesTest);
imgIdxTrain = zeros(nClasses, nSamplesTrain);
imgIdxTest = zeros(nClasses, nSamplesTest);
%% Feature detection and descriptors extraction
disp('Loading training images...')
for iClass = 1: nClasses
    subFolderName = fullfile(folderName, classList{iClass});
    % class directory
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    % randomly choose images from the class
    imgIdx{iClass} = randperm(length(imgList));
    % obtain image index (first nSamplesTrain for training, next nSamplesTest for testing)
    imgIdxTrain(iClass, :) = imgIdx{iClass}(1: nSamplesTrain);
    imgIdxTest(iClass, :) = imgIdx{iClass}(nSamplesTrain + 1: nSamplesTrain + nSamplesTest);
    for iSample = 1: nSamplesTrain
        % read image
        imgTrain = imread(fullfile(subFolderName,imgList(imgIdxTrain(iClass, iSample)).name));
        % if the image is not in gray scale
        if size(imgTrain,3) == 3
            % PHOW work on gray scale image
            imgTrain = rgb2gray(imgTrain);
        end
        % obtain training descriptors
        [~, descTrain{iClass, iSample}] = vl_phow(single(imgTrain),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
        % label the desciptors
        labelTrain = ones(1, size(descTrain{iClass, iSample}, 2)) * iClass;
        descTrain{iClass, iSample} = [descTrain{iClass, iSample}; labelTrain];
    end
%     for iSample = 1: nSamplesTest
%         % read image
%         imgTest = imread(fullfile(subFolderName,imgList(imgIdxTest(iClass, iSample)).name));
%         % if the image is not in gray scale
%         if size(imgTest,3) == 3
%             % PHOW work on gray scale image
%             imgTest = rgb2gray(imgTest);
%         end
%         % obtain testing descriptors
%         [~, descTest{iClass, iSample}] = vl_phow(single(imgTest),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
%     end
end
%% Build visual codebook
disp('Building visual codebook...')
% build visual vocabulary (codebook) for 'Bag-of-Words method': randomly select 100k SIFT descriptors for clustering
descSelect = single(vl_colsubset(cat(2,descTrain{:}), 1e4));
%% Grow the forest
forest = growTrees(descSelect', param);
nTrees = length(forest);
%% Training data: assign patch descriptors to the visual codebook (vector quantisation)
disp('Encoding training images...')
% % frequency of descriptors for train dataset (for histogram)
% freqTrain = zeros(nClasses * nSamplesTrain, nClusters);
% % figures label
% labelTrain = zeros(nClasses * nSamplesTrain, 1);
for iClass = 1: nClasses
    % get image directory
    subFolderName = fullfile(folderName, classList{iClass});
    imgList = dir(fullfile(subFolderName, '*.jpg'));
    if showImg
        figure;
        suptitle('Training image representations: 256-D histograms');
    end
    for iSample = 1: nSamplesTrain
        % update descriptors in current image
        descCurr = single(descTrain{iClass, iSample});
        % number of descriptors in current image
        nDescs = size(descCurr, 2);
        % number of clusters
        nClusters = size(forest(1).prob,1);
        % frequency of descriptors for train dataset (for histogram)
        freqTrain = zeros(nClasses * nSamplesTrain, nClusters);
        % decision on leaves
        indexTrain = zeros(nDescs, nTrees);
        for iDesc = 1: nDescs
            indexTrain(iDesc, :) = testTrees_fast(descCurr(1: end - 1, iDesc)', forest);
        end
        % compute frequency of clusters for histogram
        freqTrain((iClass - 1) * nSamplesTrain + iSample, :) = histcounts(indexTrain, nClusters) / nDescs;
        % display training images and corresponding histograms
        if showImg
            I = imread(fullfile(subFolderName, imgList(imgIdxTrain(iClass, iSample)).name));
            subplot(ceil(nSamplesTrain / 3), 6 ,2 * iSample - 1);
            imshow(I);
            subplot(ceil(nSamplesTrain / 3), 6 ,2 * iSample);
            histogram(indexTrain, nClusters);
            xlim([0 nClusters + 1]);
            drawnow;
        end
    end
    % update corresponding labels
    labelTrain((iClass - 1) * nSamplesTrain + 1: iClass * nSamplesTrain) = ones(nSamplesTrain, 1) * iClass;
end

end

