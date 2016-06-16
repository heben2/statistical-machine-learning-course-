%Given the data, the number of folds and the parameters to try on each cross 
%validation, find the minimum average 0-1 loss and return this error
%along with the optimal parameter.
function [bestParams, minError] = crossValidationSvm(Data, folds, Params)
    % addpath to the libsvm toolbox
    addpath('lib/libsvm');

    Data = sortrows(Data, size(Data, 2));
    K = cell(folds, 1);

    for i=1:size(Data, 1)
        idx = mod(i, folds) + 1;
        K{idx} = [ K{idx} ; Data(i, :) ];
    end

    Sums = zeros(size(Params, 1), 1);

    for i=1:length(K)
        TrainData = [];
        for j=1:length(K)
            if j == i
                HeldOut = K{j};
            else
                TrainData = [TrainData ; K{j}];
            end
        end

        for r = 1:size(Params, 1)
            flags = ['-c ' num2str(Params(r, 1)) ' -g ' num2str(Params(r, 2)) ' -q'];
            model = svmtrain(TrainData(:,end), TrainData(:,1:end-1), flags);
            C = svmpredict(HeldOut(:, end), HeldOut(:, 1:end-1), model, '-q');
            Sums(r) = Sums(r) + sum(HeldOut(:, end) ~= C) / length(C);
        end
    end

    [minError, idx] = min(Sums);
    minError = minError / folds;
    bestParams = Params(idx, :);
end