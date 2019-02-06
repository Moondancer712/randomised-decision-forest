function [accuracy, confMat] = classification(nClasses, data, forest, showHist)
% Function:
%   - classify the data based on random forest method
%
% InputArg(s):
%   - nClasses: number of classes of images
%   - data: vectorised data with or without label
%   - forest: random forest build on training data
%   - showHist: show histogram or not
%
% OutputArg(s):
%   - accuracy: the correct rate of classification
%   - confMat: the confusion matrix
%
% Comments:
%   - adjust parameters for better results
%
% Author & Date: Yang (i@snowztail.com) - 05 Feb 19

% number of decision trees
nTrees = length(forest);
% number of samples per class
nSamples = size(data, 1) / nClasses;
% probability that current image belongs to every classes
prob = cell(nClasses, nSamples);
% confusion matrix
confMat = zeros(nClasses);
% correctness counter
crtCounter = zeros(nClasses, 1);
for iClass = 1: nClasses
    if showHist
        figure;
        suptitle(sprintf('Estimated distribution for class %d', iClass));
    end
    for iSample = 1: nSamples
        leafIdx = testTrees_fast(data((iClass - 1) * nSamples + iSample, 1: end - 1), forest);
        % probability of corresponding leaves
        probLeaf = forest(1).prob(leafIdx, :);
        % compute the class probability based on judgements by leaves
        probCurr = sum(probLeaf) / nTrees;
        % decision on category
        [~, decIdx] = max(probCurr);
        if decIdx == iClass
            crtCounter(iClass) = crtCounter(iClass) + 1;
        end
        if showHist
            subplot(ceil(nSamples / 3), 3 , iSample);
            barHandle = bar(probCurr);
            barHandle.FaceColor = 'flat';
            barHandle.CData(decIdx, :) = [1 0 0];
            axis([0, nClasses + 0.5, 0, 1]);
            xlabel('Class index');
            ylabel('Probability');
        end
        prob{iClass, iSample} = probCurr;
        confMat(iClass, decIdx) = confMat(iClass, decIdx) + 1;
    end
end
% accuracy matrix for every class
accuMat = crtCounter / nSamples;
accuracy = mean(accuMat);
end

