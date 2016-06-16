%Computes the backpropagation based on the given in weights and out weights.
%Returns the change in weights, which is used to determine which way to descent
function [deltaInWeights, deltaOutWeights] = backPropagation(Data, h, hdiff, InWeights, OutWeights)
    noOfNodes = size(InWeights, 1);
    Datapoints = num2cell(Data(:,1:end-1), 2);
    ys = cellfun(@(xrow) neuralNetwork(h, xrow, InWeights, OutWeights), Datapoints);
    
    deltaKs = num2cell(ys - Data(:, end), 2);

    deltaInWeights = zeros(size(InWeights));
    deltaOutWeights = zeros(size(OutWeights'));

    xs = num2cell([ones(size(Data, 1), 1) Data(:,1:end-1)], 2);

    as = cellfun(@(x) arrayfun(@(k) sum(InWeights(k,:).*x), [1:noOfNodes]), xs, 'UniformOutput', false);
    zs = cellfun(@(a) [1 arrayfun(@(ai) h(ai), a)], as, 'UniformOutput', false);
    deltaJs = cellfun(@(a, deltaK) arrayfun(@(w, ai) hdiff(ai)*w*deltaK, OutWeights(2:end), a'), as, deltaKs, 'UniformOutput', false);
    deltaInWeights = cellfun(@(deltaJ, x) deltaJ * x, deltaJs, xs, 'UniformOutput', false);
    deltaInWeights = sum(reshape(cell2mat(deltaInWeights'),size(InWeights,1),size(InWeights,2),[]),3) / size(Data, 1);
    deltaOutWeights = cellfun(@(deltaK, z) deltaK * z, deltaKs, zs, 'UniformOutput', false);
    deltaOutWeights = sum(reshape(cell2mat(deltaOutWeights'),size(OutWeights,2),size(OutWeights,1),[]),3)' / size(Data, 1);
end