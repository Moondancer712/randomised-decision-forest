function label = testTrees_fast(data, tree, wlType)
% Function:
%   - obtain data label by random forest based on selected weak learner
%    type
%
% InputArg(s):
%   - data: vectorised data with or without label
%   - tree: predetermined parameters of random forest
%       - tree.splitNum: number of candidate weak learners
%       - tree.depth: number of layers
%       - tree.split: criteria in split decision (information gain, gain ratio, etc)
%       - tree.num: number of trees
%   - wlType: type of the weak learner (now support 'axis-aligned' and '2-pixel' test)
%
% OutputArg(s):
%   - label: decision of classification by trained forest
%
% Comments:
%   - original version by Kim supports axis-aligned weak learner only
%   - modified to support 2-pixel test by Yang
%   - 2-pixel test (2-D) is an approximation of gradient but with higher
%   complexity than axis-aligned (1-D)
%
% Author & Date: Yang (i@snowztail.com) - 05 Feb 19

label = zeros(1, length(tree));
idx = cell(1, length(tree));
switch wlType
    case 'axis-aligned'
        for T = 1:length(tree)
            idx{1} = 1:size(data,1);
            for n = 1:length(tree(T).node)
                isLeaf = ~tree(T).node(n).dim(1);
                if isLeaf
                    leaf_idx = tree(T).node(n).leaf_idx;
                    if ~isempty(tree(T).leaf(leaf_idx))
                        label(idx{n}',T) = tree(T).leaf(leaf_idx).label;
                    end
                    continue;
                end
                idx_left = data(idx{n},tree(T).node(n).dim) < tree(T).node(n).t;
                idx{n*2} = idx{n}(idx_left');
                idx{n*2+1} = idx{n}(~idx_left');
            end
        end
    case '2-pixel'
        for T = 1:length(tree)
            idx{1} = 1:size(data,1);
            for n = 1:length(tree(T).node)
                % dimensions in pairs, check the first entry for leaf
                isLeaf = ~tree(T).node(n).dim(1);
                if isLeaf
                    leaf_idx = tree(T).node(n).leaf_idx;
                    if ~isempty(tree(T).leaf(leaf_idx))
                        label(idx{n}',T) = tree(T).leaf(leaf_idx).label;
                    end
                    continue;
                end
                idx_left = data(idx{n},tree(T).node(n).dim(1)) - data(idx{n},tree(T).node(n).dim(2)) < tree(T).node(n).t;
                idx{n*2} = idx{n}(idx_left');
                idx{n*2+1} = idx{n}(~idx_left');
            end
        end
    otherwise
        % mode not supported yet
        error('Entered mode not supported yet.');
end

end

