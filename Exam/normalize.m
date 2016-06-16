%Normalize given data based on training data. Excludes last column in data, as
%this is expected to be the target variable.
function [normTrainData, normTestData] = normalize(trainData, testData)
	means = mean(trainData(:, 1:end-1), 1)';
	stds = std(trainData(:, 1:end-1), 0, 1)';

	normTrainData = [ fNorm(trainData(:, 1:end-1), means', stds') trainData(:, end) ];
	normTestData = [ fNorm(testData(:, 1:end-1), means', stds') testData(:, end) ];
end

function normData = fNorm(data, means, stds)
    normData = data - repmat(means, size(data,1), 1);
    normData = normData ./ repmat(stds, size(data,1), 1);
end