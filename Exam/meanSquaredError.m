function [err, ys] = meanSquaredError(Data, InWeights, OutWeights, h)
	Datapoints = num2cell(Data(:,1:end-1), 2);
    ys = cellfun(@(xrow) neuralNetwork(h, xrow, InWeights, OutWeights), Datapoints);
    err = sum((ys - Data(:, end)).^2) / size(Data, 1);
end