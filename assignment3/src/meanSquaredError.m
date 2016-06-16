function [err, ys] = meanSquaredError(Data, InWeights, OutWeights, h)
    ys = arrayfun(@(x) neuralNetwork(h, x, InWeights, OutWeights), Data(:,1));
    err = sum((ys - Data(:, 2)).^2) / size(Data, 1);
end