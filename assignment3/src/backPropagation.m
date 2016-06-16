function [deltaInWeights, deltaOutWeights] = backPropagation(Data, h, hdiff, InWeights, OutWeights)
    noOfNodes = size(InWeights, 1);
    ys = arrayfun(@(x) neuralNetwork(h, x, InWeights, OutWeights), Data(:,1));
    deltaKs = mat2cell(ys - Data(:, 2), ones(1, size(ys, 1)), 1);
    deltaInWeights = zeros(size(InWeights));
    deltaOutWeights = zeros(size(OutWeights'));
    xs = mat2cell([ones(size(Data, 1), 1) Data(:,1)], ones(1, size(Data, 1)), 2);
    as = cellfun(@(x) arrayfun(@(k) sum(InWeights(k,:).*x), [1:noOfNodes]), xs, 'UniformOutput', false);
    zs = cellfun(@(a) [1 arrayfun(@(ai) h(ai), a)], as, 'UniformOutput', false);
    deltaJs = cellfun(@(a, deltaK) arrayfun(@(w, ai) hdiff(ai)*w*deltaK, OutWeights(2:end), a'), as, deltaKs, 'UniformOutput', false);
    deltaInWeights = cellfun(@(deltaJ, x) deltaJ * x, deltaJs, xs, 'UniformOutput', false);
    deltaInWeights = sum(reshape(cell2mat(deltaInWeights'),size(InWeights,1),size(InWeights,2),[]),3) / size(Data, 1);
    deltaOutWeights = cellfun(@(deltaK, z) deltaK * z, deltaKs, zs, 'UniformOutput', false);
    deltaOutWeights = sum(reshape(cell2mat(deltaOutWeights'),size(OutWeights,2),size(OutWeights,1),[]),3)' / size(Data, 1);
end