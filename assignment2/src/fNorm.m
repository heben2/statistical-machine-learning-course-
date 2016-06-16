function normData = fNorm(data, means, stds)
    normData = data(:,1:2) - repmat(means, size(data,1), 1);
    normData = normData ./ repmat(stds, size(data,1), 1);
    normData = [ normData data(:,3) ];
end