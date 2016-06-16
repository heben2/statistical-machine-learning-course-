function [kBest, kError] = crossValidation(Data, folds, Range)
    Data = sortrows(Data, 3);
    K = cell(folds, 1);

    for i=1:size(Data, 1)
        idx = mod(i, folds) + 1;
        K{idx} = [ K{idx} ; Data(i, :) ];
    end

    Sums = zeros(size(Range));

    for i=1:length(K)
        TrainData = [];
        for j=1:length(K)
            if j == i
                HeldOut = K{j};
            else
                TrainData = [TrainData ; K{j}];
            end
        end

        for r = 1:length(Range)
            C = arrayfun(@(x, y) kNN(TrainData, [x y], Range(r)), HeldOut(:, 1), HeldOut(:, 2));
            Sums(r) = Sums(r) + sum(HeldOut(:, 3) ~= C) / length(C);
        end
    end

    [kError, idx] = min(Sums);
    kError = kError / folds;
    kBest = Range(idx);
end