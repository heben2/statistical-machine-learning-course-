%Given data and a number of folds along with the range for k, 
%do cross validation to find the minimum average 0-1 loss and return this error
%along with the optimal k.
function [kBest, kError] = crossValidationKnn(Data, folds, Range)
    %Split data into folds with roughly equaly distributed classifiers.
    Data = sortrows(Data, size(Data, 2));
    K = cell(folds, 1);

    for i=1:size(Data, 1)
        idx = mod(i, folds) + 1;
        K{idx} = [ K{idx} ; Data(i, :) ];
    end

    Sums = zeros(size(Range));

    %Crossvalidate for all k in range Range for a current setup (split) per iteration.
    for i=1:length(K)
        TrainData = [];
        for j=1:length(K)
            if j == i
                HeldOut = K{j};
            else
                TrainData = [TrainData ; K{j}];
            end
        end

        mdl = ClassificationKNN.fit(TrainData(:,1:end-1),TrainData(:,end),...
                    'BreakTies','nearest');
        for r = 1:length(Range)
            %Set number of neighbors for current test.
            mdl.NumNeighbors = Range(r);
            %The loss is average 0-1 loss.
            Y = predict(mdl,HeldOut(:,1:end-1));
            Sums(r) = Sums(r) + sum(HeldOut(:, end) ~= Y) / length(Y);
        end
    end
    %Find minimum error
    [kError, idx] = min(Sums);
    %calculate true average 0-1 loss over all splits
    kError = kError / folds;
    kBest = Range(idx);
end