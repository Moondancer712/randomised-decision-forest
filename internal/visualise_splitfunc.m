function visualise_splitfunc(idx_best,data,dim,t,target,iter) % Draw the split line
% target can be information gain and information gain ratio
r = [-1.5 1.5]; % Data range

subplot(2,2,1);
if dim == 1
    plot([t t],[r(1),r(2)],'r');
else
    plot([r(1),r(2)],[t t],'r');
end
hold on;
plot(data(~idx_best,1), data(~idx_best,2), '*', 'MarkerEdgeColor', [.8 .6 .6], 'MarkerSize', 10);
hold on;
plot(data(idx_best,1), data(idx_best,2), '+', 'MarkerEdgeColor', [.6 .6 .8], 'MarkerSize', 10);

hold on;
plot(data(data(:,end)==1,1), data(data(:,end)==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
hold on;
plot(data(data(:,end)==2,1), data(data(:,end)==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
hold on;
plot(data(data(:,end)==3,1), data(data(:,end)==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');

if ~iter
    switch target.name
        case 'IG'
            title(sprintf('BEST Split [%i]. IG = %4.2f',dim,target.value));
        case 'IGR'
            title(sprintf('BEST Split [%i]. IGR = %4.2f',dim,target.value));
        case 'Gini'
            title(sprintf('BEST Split [%i]. Gini impurity = %4.2f',dim,target.value));
        otherwise
            % mode not supported yet
            error('Entered mode not supported yet.');
    end
else
    switch target.name
        case 'IG'
            title(sprintf('Trial %i - Split [%i]. IG = %4.2f',iter,dim,target.value));
        case 'IGR'
            title(sprintf('Trial %i - Split [%i]. IGR = %4.2f',iter,dim,target.value));
        case 'Gini'
            title(sprintf('Trial %i - Split [%i]. Gini = %4.2f',iter,dim,target.value));
        otherwise
            % mode not supported yet
            error('Entered mode not supported yet.');
    end
end
axis([r(1) r(2) r(1) r(2)]);
hold off;

% histogram of base node
subplot(2,2,2);
tmp = hist(data(:,end), unique(data(:,end)));
bar(tmp);
axis([0.5 3.5 0 max(tmp)]);
title('Class histogram of parent node');
subplot(2,2,3);
bar(hist(data(idx_best,end), unique(data(:,end))));
axis([0.5 3.5 0 max(tmp)]);
title('Class histogram of left child node');
subplot(2,2,4);
bar(hist(data(~idx_best,end), unique(data(:,end))));
axis([0.5 3.5 0 max(tmp)]);
title('Class histogram of right child node');
hold off;
end
