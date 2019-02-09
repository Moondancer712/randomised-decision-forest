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
%   - modified to support 2-pixel test and linear categorisation by Yang
%   - 2-pixel test (2-D) is an approximation of gradient but with higher
%   complexity than axis-aligned (1-D)
%   - linear categorisation (2-D) is more general case of 2-pixel test,
%   where the dimensions and the parameters of functions are randomly
%   generated
%   - distributing all samples to one node can happen due to randomness of
%   parameters of linear function
%   - those nodes are invalid and can be avoid by regenerating parameters
%   controlled by var 'isSplit'
%
% Author & Date: Yang (i@snowztail.com) - 05 Feb 19

visualise = false;

% initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

% make this node a leaf if has less than 5 data points
if length(node.idx) <= 5 
    node.t = nan;
    node.dim = 0;
    return;
end

% pass data
idx = node.idx;
data = data(idx,:);
[N,D] = size(data);

% initialise the value of target functions as invalid
ig_best = -inf; 
igr_best = -inf; 
gini_best = -inf;
idx_best = [];

switch wlType
    case 'axis-aligned'
        for n = 1:iter
            % pick one random dimension
            dim = randi(D-1);
            % find the data range of this dimension
            dMin = single(min(data(:,dim))) + eps; 
            dMax = single(max(data(:,dim))) - eps;
            % pick a random value within the range as threshold
            t = dMin + rand*((dMax-dMin)); 
            % make decision
            idxCurr = data(:,dim) < t;
            
            % use different target function corresponding to evaluation
            % criteria (information gain, gain ratio, etc.)
            switch param.split
                case 'IG'
                    % calculate information gain
                    ig = getIG(data,idxCurr); 
                case 'IGR'
                    % calculate information gain ratio
                    % gain ratio can reduce the tendency of choosing nodes
                    % with more leaves
                    igr = getIGR(data, idxCurr, wlType); 
                case 'Gini'
                    % calculate Gini impurity (a low-complexity no-log
                    % approximation of information gain)
                    gini = getGini(idxCurr); 
                otherwise
                    % mode not supported yet
                    error('Entered mode not supported yet.');
            end
            
            % check that children node are not empty
            if (sum(idxCurr) > 0 && sum(~idxCurr) > 0) 
                switch param.split
                    case 'IG'
                        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idxCurr,dim,idx_best);
                    case 'IGR'
                        [node, igr_best, idx_best] = updateIGR(node,igr_best,igr,t,idxCurr,dim,idx_best);
                    case 'Gini'
                        [node, gini_best, idx_best] = updateGini(node,gini_best,gini,t,idxCurr,dim,idx_best);
                    otherwise
                        % mode not supported yet
                        error('Entered mode not supported yet.');
                end  
            end
            
        end
    case '2-pixel'
        for n = 1:iter
            % pick two random different dimension
            dim = randperm((D - 1), 2);
            % calculate differential vector
            diff = data(:, dim(1)) - data(:, dim(2));
            % get min and max value of both dimensions
            dMin = min(diff);
            dMax = max(diff);
            % pick a random value within the range as threshold
            t = dMin + rand * (dMax - dMin);
            % obtain index of left nodes
            idxCurr = diff < t;
            
            % use different target function corresponding to evaluation
            % criteria (information gain, gain ratio, etc.)
            switch param.split
                case 'IG'
                    % calculate information gain
                    ig = getIG(data,idxCurr); 
                case 'IGR'
                    % calculate information gain ratio
                    % gain ratio can reduce the tendency of choosing nodes
                    % with more leaves
                    igr = getIGR(data, idxCurr, wlType); 
                case 'Gini'
                    % calculate Gini impurity (a low-complexity no-log
                    % approximation of information gain)
                    gini = getGini(idxCurr); 
                otherwise
                    % mode not supported yet
                    error('Entered mode not supported yet.');
            end
            
            % check that children node are not empty
            if (sum(idxCurr) > 0 && sum(~idxCurr) > 0) 
                switch param.split
                    case 'IG'
                        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idxCurr,dim,idx_best);
                    case 'IGR'
                        [node, igr_best, idx_best] = updateIGR(node,igr_best,igr,t,idxCurr,dim,idx_best);
                    case 'Gini'
                        [node, gini_best, idx_best] = updateGini(node,gini_best,gini,t,idxCurr,dim,idx_best);
                    otherwise
                        % mode not supported yet
                        error('Entered mode not supported yet.');
                end  
            end
        end
    case 'linear'
        for n = 1:iter
            % ensure successful split
            isSplit = false;
            while ~isSplit
                % pick two random different dimension
                dim = randperm((D - 1), 2);
                % coefficients of linear function
                t = randn(3, 1);
                % decision
                idxCurr = [data(:, dim), ones(N, 1)] * t < 0;
                isSplit = ~isequal(idxCurr, zeros(size(idxCurr))) && ~isequal(idxCurr, ones(size(idxCurr)));
            end
            
            % use different target function corresponding to evaluation
            % criteria (information gain, gain ratio, etc.)
            switch param.split
                case 'IG'
                    % calculate information gain
                    ig = getIG(data,idxCurr); 
                case 'IGR'
                    % calculate information gain ratio
                    % gain ratio can reduce the tendency of choosing nodes
                    % with more leaves
                    igr = getIGR(data, idxCurr, wlType); 
                case 'Gini'
                    % calculate Gini impurity (a low-complexity no-log
                    % approximation of information gain)
                    gini = getGini(idxCurr); 
                otherwise
                    % mode not supported yet
                    error('Entered mode not supported yet.');
            end
            
            % check that children node are not empty
            if (sum(idxCurr) > 0 && sum(~idxCurr) > 0) 
                switch param.split
                    case 'IG'
                        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idxCurr,dim,idx_best);
                    case 'IGR'
                        [node, igr_best, idx_best] = updateIGR(node,igr_best,igr,t,idxCurr,dim,idx_best);
                    case 'Gini'
                        [node, gini_best, idx_best] = updateGini(node,gini_best,gini,t,idxCurr,dim,idx_best);
                    otherwise
                        % mode not supported yet
                        error('Entered mode not supported yet.');
                end  
            end
            
        end
    otherwise
        % mode not supported yet
        error('Entered mode not supported yet.');
end
% update node index
nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

% not supported for 'Caltech' data yet
if visualise
    target.name = param.split;
    switch param.split
        case 'IG'
            target.value = ig_best;
            fprintf('Information gain = %f. \n',ig_best);
        case 'IGR'
            target.value = igr_best;
            fprintf('Information gain ratio = %f. \n',igr_best);
        case 'Gini'
            target.value = gini_best;
            fprintf('Gini impurity = %f. \n',gini_best);
        otherwise
            % mode not supported yet
            error('Entered mode not supported yet.');
    end
    visualise_splitfunc(idx_best,data,dim,t,target,0);
    pause();
end

end

% information gain: the 'purity' of data labels in both child nodes after 
% split. The higher the purer.
function ig = getIG(data,idx) 
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

% information gain ratio: a ratio of information gain to the intrinsic 
% information. Notice the RANDOMNESS in computing the intrinsic information
% of the next layer will average to ground truth by RF and LLN.
function igr = getIGR(data, idx, wlType) 
% compute information gain
ig = getIG(data,idx);
[N,D] = size(data);
% classification: one step forward to compute the node entropy
switch wlType
    case 'axis-aligned'
        % pick one random dimension
        dim = randi(D-1);
        % find the data range of this dimension
        dMin = single(min(data(:,dim))) + eps;
        dMax = single(max(data(:,dim))) - eps;
        % pick a random value within the range as threshold
        t = dMin + rand*((dMax-dMin));
        % obtain index of left nodes of the layer after
        idxNext = data(:,dim) < t;
    case '2-pixel'
        % pick two random different dimension
        dim = randperm((D - 1), 2);
        % calculate differential vector
        diff = data(:, dim(1)) - data(:, dim(2));
        % get min and max value of both dimensions
        dMin = min(diff);
        dMax = max(diff);
        % pick a random value within the range as threshold
        t = dMin + rand * (dMax - dMin);
        % obtain index of left nodes of the layer after
        idxNext = diff < t;
    case 'linear'
        % ensure successful split
        isSplit = false;
        while ~isSplit
            % pick two random different dimension
            dim = randperm((D - 1), 2);
            % coefficients of linear function
            t = randn(3, 1);
            % obtain index of left nodes of the layer after
            idxNext = [data(:, dim), ones(N, 1)] * t < 0;
            isSplit = ~isequal(idxNext, zeros(size(idxNext))) && ~isequal(idxNext, ones(size(idxNext)));
        end
    otherwise
        % mode not supported yet
        error('Entered mode not supported yet.');
end
% probabilities of next layer to go left or right
probL = sum(idxNext) / length(idxNext);
probR = 1 - probL;
% use probabilities to compute entropy of the NODE (not the data)
H = -log2(probL) * probL - log2(probR) * probR;
% compute information gain ratio
igr = ig / H;
end

% Gini impurity is a measure of how often a randomly chosen element from 
% the set would be incorrectly labeled if it was randomly labeled according
% to the distribution of labels in the subset (a low-complexity no-log 
% approximation of information gain)
% ratio)
function gini = getGini(idx)
% probabilities of the current layer to go left or right
probL = sum(idx) / length(idx);
probR = 1 - probL;
% compute Gini impurity
gini = 1 - probL ^ 2 - probR ^ 2;
end

% entropy
function H = getE(X) 
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

% update information gain
function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) 
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
end
end

% Update information gain ratio
function [node, igr_best, idx_best] = updateIGR(node,igr_best,igr,t,idx,dim,idx_best) 
if igr > igr_best
    igr_best = igr;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
end
end

% Update Gini impurity
function [node, gini_best, idx_best] = updateGini(node,gini_best,gini,t,idx,dim,idx_best)
if gini > gini_best
    gini_best = gini;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
end
end
