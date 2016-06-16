function wML = linearRegression(data)
    Phi = [ones(size(data, 1), 1) data(:, 1:end-1)];
    wML = pinv(Phi) * data(:, end);
end