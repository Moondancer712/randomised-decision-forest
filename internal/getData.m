function [data_train, data_query] = getData(MODE, nClusters, showImg, showHist)
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 1; % Show training & testing images and their image feature vector (histogram representation)

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
        close all;
        %% Feature detection and descriptors extraction
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name}; % 10 classes
        % number of classes
        nClasses = length(classList);
        % number of samples per class
        nSamples = imgSel(1);
        % initialisation
        imgIdx = cell(nClasses, 1);
        descTrain = cell(nClasses, nSamples);
%         descTest = cell(nClasses, nSamples);
        imgIdxTrain = zeros(nClasses, nSamples);
        imgIdxTest = zeros(nClasses, nSamples);
        
        disp('Loading training images...')
        
        for iClass = 1: nClasses
            subFolderName = fullfile(folderName, classList{iClass});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{iClass} = randperm(length(imgList));
            imgIdxTrain(iClass, :) = imgIdx{iClass}(1: nSamples);
            imgIdxTest(iClass, :) = imgIdx{iClass}(nSamples + 1: sum(imgSel));
            
            for iSample = 1: nSamples
                imgTrain = imread(fullfile(subFolderName,imgList(imgIdxTrain(iClass, iSample)).name)); 
%                 imgTest = imread(fullfile(subFolderName,imgList(imgIdxTest(iClass, iSample)).name)); 

                % if the image is not in gray scale
                if size(imgTrain,3) == 3
                    imgTrain = rgb2gray(imgTrain); % PHOW work on gray scale image
                end
%                 if size(imgTest,3) == 3
%                     imgTest = rgb2gray(imgTest); % PHOW work on gray scale image
%                 end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, descTrain{iClass, iSample}] = vl_phow(single(imgTrain),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
%                 [~, descTest{iClass, iSample}] = vl_phow(single(imgTest),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        descSelect = single(vl_colsubset(cat(2,descTrain{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        
        %% K-means clustering
        % index: the cluster index that descripters belongs to
        % centroid: the centroids position
%         [~, centroid] = kmeans(descSelect', nClusters);
        [centroid, ~] = vl_kmeans(descSelect, nClusters);
        centroid = centroid';
        %% Histograms
        disp('Encoding Images...')
        % frequency of descriptors for train dataset
        freqTrain = cell(nClasses, nSamples);
        % Assign patch descriptors to the visual codebook
        for iClass = 1: nClasses
            subFolderName = fullfile(folderName, classList{iClass});
            imgList = dir(fullfile(subFolderName, '*.jpg'));
            if showImg
                % image counter
                counter = 0;
                figure;
            end
            for iSample = 1: nSamples
                % update current descriptor
                descCurr = single(descTrain{iClass, iSample});
                % categorise by KNN based on obtained centroids
                indexTrain = knnsearch(centroid, descCurr');
                % compute frequency of centroids for histogram
                freqTrain{iClass, iSample} = histcounts(indexTrain, nClusters) / length(indexTrain);
                % display train images and corresponding histograms
                if showImg
                    counter = counter + 1;
                    I = imread(fullfile(subFolderName, imgList(imgIdxTrain(iClass, iSample)).name));
%                     subaxis(2, nSamples, counter, 'SpacingVert', 0, 'MR', 0);
%                     subplot(nSamples, 2, 2 * iSample - 1);
                    subplot(ceil(nSamples / 3), 6 ,2 * iSample - 1);
                    imshow(I);
%                     subaxis(2, nSamples, nSamples + counter, 'SpacingVert', 0, 'MR', 0);
%                     subplot(nSamples, 2, 2 * iSample);
                    subplot(ceil(nSamples / 3), 6 ,2 * iSample);
                    histogram(indexTrain, nClusters);
                    xlim([0 nClusters + 1]);
                    drawnow;
                end
            end
            pause;
        end
        % Vector Quantisation
        
        % write your own codes here
        % ...
  
        
        % Clear unused varibles to save memory
        clearvars descTrain descSelect
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for iClass = 1:length(classList)
            subFolderName = fullfile(folderName,classList{iClass});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdxTest = imgIdx{iClass}(imgSel(1)+1:sum(imgSel));
            
            for iSample = 1:length(imgIdxTest)
                I = imread(fullfile(subFolderName,imgList(imgIdxTest(iSample)).name));
                
                % Visualise
                if iSample < 6 && showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{iClass,iSample}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        %suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

        % Quantisation
        
        % write your own codes here
        % ...
        
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

