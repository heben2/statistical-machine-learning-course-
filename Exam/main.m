%Note that I have hardcoded a random seed into the assignments to produce same 
%results as the assignment is build on.

%%
% Question 1
%
clear;fprintf('############### Question 1 ###############\n');
trainData = importdata('../data/SSFRTrain2014.dt');
testData = importdata('../data/SSFRTest2014.dt');

%Maximum Likelihood
wML = linearRegression(trainData)
y = @(x, wML) x * wML;
trainML = arrayfun(@(x1, x2, x3, x4) y([1 x1 x2 x3 x4], wML), trainData(:,1), ...
              trainData(:,2), trainData(:,3), trainData(:,4));
testML = arrayfun(@(x1, x2, x3, x4) y([1 x1 x2 x3 x4], wML), testData(:,1), ...
              testData(:,2), testData(:,3), testData(:,4));

trainMSE = 1/size(trainData,1)*sum((trainData(:,5)-trainML).^2)
testMSE = 1/size(testData,1)*sum((testData(:,5)-testML).^2)

%%
% Question 2
%
clear;fprintf('############### Question 2 ###############\n');
trainData = importdata('../data/SSFRTrain2014.dt');
testData = importdata('../data/SSFRTest2014.dt');

%chose sigmoid as activation function
h = @(a) a / (1 + abs(a));
hdiff = @(a) 1 / (1 + abs(a))^2;

randomSeed = rng(43786953);
dimInput = 4;
StartInWeights = random('unif', 0, 1, 40, dimInput + 1);
StartOutWeights = random('unif', 0, 1, 41, 1);

largeLearningRate = 0.1;
smallLearningRate = 0.001;

%Best found network is with 40 hidden nodes, @smallLearningRate = 0.001, largeLearningRate = 0.1
%WARNING: LONG RUNNING TIME, approximately 30 minutes.
[InWeights, OutWeights, TrainError10Nodes, TestError10Nodes] ...
	= steepestDescent(trainData, testData, StartInWeights, ...
      StartOutWeights, h, hdiff, 10E-5, largeLearningRate, smallLearningRate);
fprintf('40 random hidden nodes.\nLarge learning rate: %.3f\nSmall learning rate: %.3f\nTraining error = %.5f\nTest error = %.5f\n', ...
	largeLearningRate, smallLearningRate, TrainError10Nodes(end), ...
	TestError10Nodes(end));


%%
% Question 3
%
clear;fprintf('############### Question 3 ###############\n');
trainData = importdata('../data/SGTrain2014.dt');
testData = importdata('../data/SGTest2014.dt');
% addpath to the libsvm toolbox
addpath('lib/libsvm');
format long

%Normalize data
[normTrainData, normTestData] = normalize(trainData, testData);

%Use Jaakkola heuristic to determine gamma
yJaakkola = gammaJaakkola(trainData)
b = 10; %should be either 2 or 10
%Init grid
n1 = [-2:3];
n2 = [-3:3];
i = b.^n1' * ones(size(n2));
j = yJaakkola*b.^n2' * ones(size(n2));
j(1:end-7)';
params = [ i(:) j(1:end-7)'];

% first column of bestParams = cost, second column of bestParams = gamma
[normBestParams, normMinError] = crossValidationSvm(normTrainData, 5, params);
fprintf('Best found C = %6.4f, gamma = %6.4f\n',normBestParams(1), normBestParams(2));
normflags = ['-c ' num2str(normBestParams(1)) ' -g ' num2str(normBestParams(2)) ' -q'];
normModel = svmtrain(normTrainData(:,end), normTrainData(:,1:end-1), normflags);
fprintf('Train accuracy:')
normC = svmpredict(normTrainData(:, end), normTrainData(:, 1:end-1), normModel);
fprintf('Test accuracy:')
normC = svmpredict(normTestData(:, end), normTestData(:, 1:end-1), normModel);


%%
% Question 4
%
clear;fprintf('############### Question 4 ###############\n');
trainData = importdata('../data/SGTrain2014.dt');
testData = importdata('../data/SGTest2014.dt');

%Normalize data
[normTrainData, normTestData] = normalize(trainData, testData);
trainDataGalaxies = normTrainData(normTrainData(:,end) == 0, :);
trainDataGalaxies = trainDataGalaxies(:,1:end-1);

%Compute eigenvectors and eigenvalues, and order them in descending order of eigenvalue
[EigenVectors, eigenValues] = eig(cov(trainDataGalaxies));
[eigenValues order] = sort(diag(eigenValues), 'descend');
EigenVectors = EigenVectors(:,order);

%plot of eigenspectrum
h = figure(4);
plot(eigenValues,[1:length(eigenValues)],'b-');
ylabel('Lambda_i');
xlabel('i');
betterPlots(h);
print(h, '-depsc2', '../figures/question4_1.eps');

%Plot of data projected onto first two principal components of PCA
pca2trainDataGalaxiesX = trainDataGalaxies*EigenVectors(:,1);
pca2trainDataGalaxiesY = trainDataGalaxies*EigenVectors(:,2);
h = figure(2);
scatter(pca2trainDataGalaxiesX,pca2trainDataGalaxiesY,'g');
xlabel('1. PC');
ylabel('2. PC');
betterPlots(h);
print(h, '-depsc2', '../figures/question4_2.eps');


%%
% Question 5
%
fprintf('############### Question 5 ###############\n');
k = 2; %number of clusters
randomSeed = rng(43786953);
%Randomly chose k datapoints from training data as initial centroids.
init_cents = trainDataGalaxies(randsample(size(trainDataGalaxies,1),k),:);
[clusterCenters, clusterSets] = k_mean(init_cents, trainDataGalaxies);
fprintf('Found 2-mean cluster centers:')
clusterCenters'

%Project cluster centers onto first two principle components
pca2trainDataClusterCentersX = clusterCenters*EigenVectors(:,1);
pca2trainDataClusterCentersY = clusterCenters*EigenVectors(:,2);

%Plot with colored clusters and cluster centers
pca2trainDataCluster1X = clusterSets{1}*EigenVectors(:,1);
pca2trainDataCluster1Y = clusterSets{1}*EigenVectors(:,2);
pca2trainDataCluster2X = clusterSets{2}*EigenVectors(:,1);
pca2trainDataCluster2Y = clusterSets{2}*EigenVectors(:,2);
h = figure(5);
scatter(pca2trainDataCluster1X,pca2trainDataCluster1Y,'g');
hold on
scatter(pca2trainDataCluster2X,pca2trainDataCluster2Y,'r');
scatter(pca2trainDataClusterCentersX,pca2trainDataClusterCentersY,'k+');
xlabel('1. PC');
ylabel('2. PC');
betterPlots(h);
print(h, '-depsc2', '../figures/question5_2.eps');
hold off


%%
% Question 7
%
clear;fprintf('############### Question 7 ###############\n');
trainData = importdata('../data/VSTrain2014.dt');
testData = importdata('../data/VSTest2014.dt');

fprintf('LDA-model')
%Linear classification: LDA
%Note that we do not need to use the normalized data because LDA is ineffected by it.
obj = ClassificationDiscriminant.fit(trainData(:,1:end-1),trainData(:,end), 'DiscrimType','linear');
trainError = loss(obj, trainData(:,1:end-1),trainData(:,end))
testError = loss(obj,testData(:,1:end-1),testData(:,end))

fprintf('Knn-model')
%Normalize data
[normTrainData, normTestData] = normalize(trainData, testData);

%Non-linear classification: k-nearest neighbor
[kBest, AvgTrainError] = crossValidationKnn(normTrainData, 5, [1:25])
mdl = ClassificationKNN.fit(normTrainData(:,1:end-1),normTrainData(:,end), 'BreakTies','nearest');
mdl.NumNeighbors = kBest;

testError = loss(mdl,normTestData(:,1:end-1),normTestData(:,end), 'lossfun','classiferror')
trainError = loss(mdl,normTrainData(:,1:end-1),normTrainData(:,end), 'lossfun','classiferror')

