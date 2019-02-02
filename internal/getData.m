function [data_train, data_query] = getData(MODE, nClusters, showImg)
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

% showImg = 1; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4;
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8;
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        %% Training data: Feature detection and descriptors extraction
        % randomly select 15 images each class without replacement. (For both training & testing)
        imgSel = [15 15]; 
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        % choose classes
        classList = {classList(3:end).name};
        % number of classes
        nClasses = length(classList);
        % number of samples per class
        nSamples = imgSel(1);
        % initialisation
        imgIdx = cell(nClasses, 1);
        descTrain = cell(nClasses, nSamples);
        imgIdxTrain = zeros(nClasses, nSamples);
        
        disp('Loading training images...')
        for iClass = 1: nClasses
            subFolderName = fullfile(folderName, classList{iClass});
            % class directory
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            % randomly choose images from the class
            imgIdx{iClass} = randperm(length(imgList));
            % obtain image index
            imgIdxTrain(iClass, :) = imgIdx{iClass}(1: nSamples);
            for iSample = 1: nSamples
                % read image
                imgTrain = imread(fullfile(subFolderName,imgList(imgIdxTrain(iClass, iSample)).name));
                % if the image is not in gray scale
                if size(imgTrain,3) == 3
                    % PHOW work on gray scale image
                    imgTrain = rgb2gray(imgTrain); 
                end
                % obtain training descriptors
                [~, descTrain{iClass, iSample}] = vl_phow(single(imgTrain),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method': randomly select 100k SIFT descriptors for clustering
        descSelect = single(vl_colsubset(cat(2,descTrain{:}), 1e4)); 
        %% K-means clustering
        % compute the centroids position
%         [~, centroid] = kmeans(descSelect', nClusters);
        [centroid, ~] = vl_kmeans(descSelect, nClusters);
        centroid = centroid';
        %% Training data: assign patch descriptors to the visual codebook (vector quantisation)
        disp('Encoding training images...')
        % frequency of descriptors for train dataset (for histogram)
        freqTrain = zeros(nClasses * nSamples, nClusters);
        % figures label
        labelTrain = zeros(nClasses * nSamples, 1);
        for iClass = 1: nClasses
            % get image directory
            subFolderName = fullfile(folderName, classList{iClass});
            imgList = dir(fullfile(subFolderName, '*.jpg'));
            if showImg
                figure;
                suptitle('Training image representations: 256-D histograms');
            end
            for iSample = 1: nSamples
                % update current descriptor
                descCurr = single(descTrain{iClass, iSample});
                % categorise by KNN based on obtained centroids
                indexTrain = knnsearch(centroid, descCurr');
                % compute frequency of centroids for histogram
                freqTrain((iClass - 1) * nSamples + iSample, :) = histcounts(indexTrain, nClusters) / length(indexTrain);
                % display training images and corresponding histograms
                if showImg
                    I = imread(fullfile(subFolderName, imgList(imgIdxTrain(iClass, iSample)).name));
                    subplot(ceil(nSamples / 3), 6 ,2 * iSample - 1);
                    imshow(I);
                    subplot(ceil(nSamples / 3), 6 ,2 * iSample);
                    histogram(indexTrain, nClusters);
                    xlim([0 nClusters + 1]);
                    drawnow;
                end
            end
            % update corresponding labels
            labelTrain((iClass - 1) * nSamples + 1: iClass * nSamples) = ones(nSamples, 1) * iClass;
        end
        % label the training data
        data_train = [freqTrain, labelTrain];
        % clear unused varibles to save memory
        clearvars descTrain descSelect
end
%% Testing data: Feature detection and descriptors extraction
switch MODE
    case 'Caltech'
        % initialisation
        descTest = cell(nClasses, nSamples);
        imgIdxTest = zeros(nClasses, nSamples);
        
        disp('Loading testing images...')
        for iClass = 1: nClasses
            subFolderName = fullfile(folderName, classList{iClass});
            % class directory
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            % randomly choose images from the class
            imgIdx{iClass} = randperm(length(imgList));
            % obtain image index
            imgIdxTest(iClass, :) = imgIdx{iClass}(nSamples + 1: sum(imgSel));
            for iSample = 1: nSamples
                % read image
                imgTest = imread(fullfile(subFolderName,imgList(imgIdxTest(iClass, iSample)).name));
                % if the image is not in gray scale
                if size(imgTest,3) == 3
                    % PHOW work on gray scale image
                    imgTest = rgb2gray(imgTest); 
                end
                % obtain testing descriptors
                [~, descTest{iClass, iSample}] = vl_phow(single(imgTest),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        %% Testing data: assign patch descriptors to the visual codebook (vector quantisation)
        disp('Analysing testing images...')
        % frequency of descriptors for testing dataset (for histogram)
        freqTest = zeros(nClasses * nSamples, nClusters);
        % Assign patch descriptors to the visual codebook (vector quantisation)
        for iClass = 1: nClasses
            subFolderName = fullfile(folderName, classList{iClass});
            % get image directory
            imgList = dir(fullfile(subFolderName, '*.jpg'));
            if showImg
                figure;
                suptitle('Testing image representations: 256-D histograms');
            end
            for iSample = 1: nSamples
                % update current descriptor
                descCurr = single(descTest{iClass, iSample});
                % categorise by KNN based on obtained centroids
                indexTest = knnsearch(centroid, descCurr');
                % compute frequency of centroids for histogram
                freqTest((iClass - 1) * nSamples + iSample, :) = histcounts(indexTest, nClusters) / length(indexTest);
                % display testing images and corresponding histograms
                if showImg
                    I = imread(fullfile(subFolderName, imgList(imgIdxTest(iClass, iSample)).name));
                    subplot(ceil(nSamples / 3), 6 ,2 * iSample - 1);
                    imshow(I);
                    subplot(ceil(nSamples / 3), 6 ,2 * iSample);
                    histogram(indexTest, nClusters);
                    xlim([0 nClusters + 1]);
                    drawnow;
                end
            end
        end
        % label the testing data (with zeros)
        data_query = [freqTest, zeros(nClasses * nSamples, 1)];
        % clear unused varibles to save memory
        clearvars descTest descSelect
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

