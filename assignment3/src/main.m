%%
% III.1.1
%
clear;fprintf('############### III.1.1 ###############\n');

trainData = importdata('../data/sincTrain25.dt');
testData = importdata('../data/sincValidate10.dt');

h = @(a) a / (1 + abs(a));
hdiff = @(a) 1 / (1 + abs(a))^2;

[deltaInWeights, deltaOutWeights] = backPropagation(trainData, h, hdiff, [1 1; 1 1], [1 1 1]')

[numDeltaInWeights, numDeltaOutWeights] = numericalDiffs(trainData, h, [1 1; 1 1], [1 1 1]', 10E-8)

%%
% III.1.2
%
clear;fprintf('############### III.1.2 ###############\n');

trainData = importdata('../data/sincTrain25.dt');
testData = importdata('../data/sincValidate10.dt');

h = @(a) a / (1 + abs(a));
hdiff = @(a) 1 / (1 + abs(a))^2;

randomSeed = rng(43786953);
StartInWeights = random('unif', 0, 1, 20, 2);
StartOutWeights = random('unif', 0, 1, 21, 1);

smallLearningRate = 0.0001
largeLearningRate = 0.1
middleLearningRate = 0.01

[InWeights1, OutWeights1, TrainError2NodesLargeRate, TestError2NodesLargeRate] = ...
    steepestDescent(trainData, testData, StartInWeights(1:2,1:2), ...
    StartOutWeights(1:3), h, hdiff, 10E-5, largeLearningRate);

[InWeights2, OutWeights2, TrainError20NodesLargeRate, TestError20NodesLargeRate] = ...
    steepestDescent(trainData, testData, StartInWeights, ...
    StartOutWeights, h, hdiff, 10E-5, largeLearningRate);

[InWeights3, OutWeights3, TrainError2NodesSmallRate, TestError2NodesSmallRate] = ...
    steepestDescent(trainData, testData, StartInWeights(1:2,1:2), ...
    StartOutWeights(1:3), h, hdiff, 10E-5, smallLearningRate);

[InWeights4, OutWeights4, TrainError20NodesSmallRate, TestError20NodesSmallRate] = ...
    steepestDescent(trainData, testData, StartInWeights, ...
    StartOutWeights, h, hdiff, 10E-5, smallLearningRate);

[InWeights5, OutWeights5, TrainError20NodesGoodRate, TestError20NodesGoodRate] = ...
    steepestDescent( trainData, testData, StartInWeights, ...
    StartOutWeights, h, hdiff, 10E-5, middleLearningRate);

handle = figure(1);
semilogy(TrainError2NodesSmallRate(:,:), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(TestError2NodesSmallRate(:,:), 'b--', 'LineWidth', 1.5);
semilogy(TrainError20NodesSmallRate(:,:), 'r-', 'LineWidth', 1.5);
semilogy(TestError20NodesSmallRate(:,:), 'r--', 'LineWidth', 1.5);
title('Small learning rate');
legend('Training data, 2 hn', 'Test data, 2 hn', ...
    'Training data, 20 hn', 'Test data, 20 hn');
ylabel('MSE');
xlabel('Learning epoch');
betterPlots(handle);
print(handle, '-depsc2', '../figures/III12_1.eps');
hold off;

handle = figure(2);
semilogy(TrainError2NodesLargeRate(:,:), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(TestError2NodesLargeRate(:,:), 'b--', 'LineWidth', 1.5);
semilogy(TrainError20NodesLargeRate(:,:), 'r-', 'LineWidth', 1.5);
semilogy(TestError20NodesLargeRate(:,:), 'r--', 'LineWidth', 1.5);
title('Large learning rate');
legend('Training data, 2 hn', 'Test data, 2 hn', ...
	'Training data, 20 hn', 'Test data, 20 hn');
ylabel('MSE');
xlabel('Learning epoch');
betterPlots(handle);
print(handle, '-depsc2', '../figures/III12_2.eps');
hold off;

handle = figure(3);
semilogy(TrainError2NodesSmallRate(:,:), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(TestError2NodesSmallRate(:,:), 'b--', 'LineWidth', 1.5);
title('Middle learning rate');
legend('Training data, 20 hn', 'Test data, 20 hn');
ylabel('MSE');
xlabel('Learning epoch');
betterPlots(handle);
print(handle, '-depsc2', '../figures/III12_3.eps');
hold off;

handle = figure(4);
sinc = @(x) sin(x) / x;

fplot(sinc, [-10 10], 'b')
hold on;
fplot(@(x) neuralNetwork(h, x, InWeights1, OutWeights1), [-10 10], 'r');
fplot(@(x) neuralNetwork(h, x, InWeights2, OutWeights2), [-10 10], 'g');
fplot(@(x) neuralNetwork(h, x, InWeights3, OutWeights3), [-10 10], 'c');
fplot(@(x) neuralNetwork(h, x, InWeights4, OutWeights4), [-10 10], 'm');
fplot(@(x) neuralNetwork(h, x, InWeights5, OutWeights5), [-10 10], 'k');
legend('sinc(x)', '2 hn, large rate', '20 hn, large rate', ...
    '2 hn, small rate', '20 hn, small rate', ...
    '20 hn, middle rate');
betterPlots(handle);
print(handle, '-depsc2', '../figures/III12_4.eps');
hold off;

%%
% III.2.1
%
clear;fprintf('############### III.2.1 ###############\n');

trainData = importdata('../data/parkinsonsTrainStatML.dt');
testData = importdata('../data/parkinsonsTestStatML.dt');

means = mean(trainData(:, 1:end-1), 1)'
vars = var(trainData(:, 1:end-1), 0, 1)'
stds = std(trainData(:, 1:end-1), 0, 1)';

normTrainData = [ fNorm(trainData(:, 1:end-1), means', stds') trainData(:, end) ];
normTestData = [ fNorm(testData(:, 1:end-1), means', stds') testData(:, end) ];

normTrainMeans =  mean(normTrainData(:, 1:end-1), 1)'
normTainVars = var(normTrainData(:, 1:end-1), 0, 1)'

normTestMeans =  mean(normTestData(:, 1:end-1), 1)'
normTestVars = var(normTestData(:, 1:end-1), 0, 1)'

%%
% III.2.2
%
fprintf('############### III.2.2 ###############\n');

% addpath to the libsvm toolbox
addpath('lib/libsvm');

n = 3;
i = 10.^[-n:n]' * ones(1,2*n+1);
j = i';
params = [ i(:) j(:) ];

% first column of bestParams = cost, second column of bestParams = gamma
[bestParams, minError] = crossValidation(trainData, 5, params)

flags = ['-c ' num2str(bestParams(1)) ' -g ' num2str(bestParams(2)) ' -q'];
model = svmtrain(trainData(:,end), trainData(:,1:end-1), flags);
C = svmpredict(testData(:, end), testData(:, 1:end-1), model);
testError = sum(testData(:, end) ~= C) / length(C)

[normBestParams, normMinError] = crossValidation(normTrainData, 5, params)

normflags = ['-c ' num2str(normBestParams(1)) ' -g ' num2str(normBestParams(2)) ' -q'];
normModel = svmtrain(normTrainData(:,end), normTrainData(:,1:end-1), normflags);
normC = svmpredict(normTestData(:, end), normTestData(:, 1:end-1), normModel);
normTestError = sum(normTestData(:, end) ~= normC) / length(normC)

%%
% III.2.3.1
%
fprintf('############### III.2.3.1 ###############\n');

totalSV = getfield(model, 'totalSV')
boundedSV = sum(abs(getfield(model, 'sv_coef')) == bestParams(1))
freeSV = totalSV - boundedSV

changeC = 3;

flagsSmallC = ['-c ' num2str(bestParams(1)*10^-changeC) ' -g ' num2str(bestParams(2)) ' -q'];
modelSmallC = svmtrain(trainData(:,end), trainData(:,1:end-1), flagsSmallC);

totalSVSmallC = getfield(modelSmallC, 'totalSV')
boundedSVSmallC = sum(abs(getfield(modelSmallC, 'sv_coef')) == bestParams(1)*10^-changeC)
freeSVSmallC = totalSVSmallC - boundedSVSmallC

flagsLargeC = ['-c ' num2str(bestParams(1)*10^changeC) ' -g ' num2str(bestParams(2)) ' -q'];
modelLargeC = svmtrain(trainData(:,end), trainData(:,1:end-1), flagsLargeC);

totalSVLargeC = getfield(modelLargeC, 'totalSV')
boundedSVLargeC = sum(abs(getfield(modelLargeC, 'sv_coef')) == bestParams(1)*10^changeC)
freeSVLargeC = totalSVLargeC - boundedSVLargeC