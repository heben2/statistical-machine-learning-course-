function resClass = kNN(trainingset, dataPoint, k)
    X = trainingset(:, 1);
    Y = trainingset(:, 2);
    C = trainingset(:, 3);
    x = dataPoint(1);
    y = dataPoint(2);
    D = sqrt((X - x).^2 + (Y - y).^2);
    DC = sortrows([D C], 1);
    DC = DC(1:k, 2);
    
    % Finds the most frequent class.
    % Ties are broken by choosing the smallest
    resClass = mode(DC);
end