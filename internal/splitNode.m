function [node, nodeL, nodeR] = splitNode(data, node, param, wlType)
% Function:
%   - generate visual codebook by randomised decision forest
%
% InputArg(s):
%   - data: vectorised data with or without label
%   - param: predetermined parameters of random forest
%       - param.splitNum: number of candidate weak learners
%       - param.depth: number of layers
%       - param.split: criteria in split decision (information gain, gain ratio, etc)
%       - param.num: number of trees
%   - node: node of random forest
%       - node.idx: data (index only) which split into this node
%       - node.t: threshold of split function
%       - node.dim: feature dimension of split function
%       - node.prob: class distribution of data in this node
%       - node.leaf_idx: leaf node index (empty if it is not a leaf node)
%   - wlType: type of the weak learner (now support 'axis-aligned' and '2-pixel' test)
%
% OutputArg(s):
%   - node: node of random forest
%   - nodeL: the left subnode
%   - nodeR: the right subnode
%
% Comments:
%   - original version by Kim supports axis-aligned weak learner only
%   - modified to support 2-pixel test by Yang
%   - 2-pixel test (2-D) is an approximation of gradient but with higher
%   complexity than axis-aligned (1-D)
%
% Author & Date: Yang (i@snowztail.com) - 05 Feb 19

visualise = 0;

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[~,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];

switch wlType
    case 'axis-aligned'
        for n = 1:iter
            dim = randi(D-1); % Pick one random dimension
            d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
            d_max = single(max(data(:,dim))) - eps;
            t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
            idx_ = data(:,dim) < t;
            
            ig = getIG(data,idx_); % Calculate information gain
            
            if visualise
                visualise_splitfunc(idx_,data,dim,t,ig,n);
                pause();
            end
            
            if (sum(idx_) > 0 && sum(~idx_) > 0) % We check that children node are not empty
                [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
            end
            
        end
    case '2-pixel'
        for n = 1:iter
            % pick two random different dimension
            dim = randperm((D - 1), 2);
%             % get min and max value of both dimensions
%             dMin1 = single(min(data(:, dim(1)))) + eps;
%             dMin2 = single(min(data(:, dim(2)))) + eps;
%             dMax1 = single(max(data(:, dim(1)))) - eps;
%             dMax2 = single(max(data(:, dim(2)))) - eps;
%             % find range of subtraction
%             dMin = min(dMin1, dMin2);
%             dMax = max(dMax1, dMax2);
%             % pick a random value within the range as threshold
%             t = dMin + rand * (dMax - dMin);
%             % obtain index of left nodes
%             idx_ = data(:, dim(1)) - data(:, dim(2)) < t;
            diff = data(:, dim(1)) - data(:, dim(2));
            % get min and max value of both dimensions
            dMin = min(diff);
            dMax = max(diff);
            % pick a random value within the range as threshold
            t = dMin + rand * (dMax - dMin);
            % obtain index of left nodes
            idx_ = data(:, dim(1)) - data(:, dim(2)) < t;
            
            ig = getIG(data,idx_); % Calculate information gain
            
            if visualise
                visualise_splitfunc(idx_,data,dim,t,ig,n);
                pause();
            end
            
            if (sum(idx_) > 0 && sum(~idx_) > 0) % We check that children node are not empty
                [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
            end
            
        end
    otherwise
        % mode not supported yet
        error('Entered mode not supported yet.');
end
nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
% else
%     idx_best = idx_best;
end
end
