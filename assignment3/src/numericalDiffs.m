function [deltaInWeights, deltaOutWeights] = numericalDiffs(Data, h, InWeights, OutWeights, epsilon)
    E = meanSquaredError(Data, InWeights, OutWeights, h);

    e = eye(size(InWeights, 1));
    deltaInWeights = zeros(size(InWeights, 1), 1);
    for j = 1:size(InWeights, 1)
        InWeightsEpsilon = InWeights + repmat(epsilon*e(:,j),1,size(InWeights,2));
        Eepsilon = meanSquaredError(Data, InWeightsEpsilon, OutWeights, h);
        deltaInWeights(j,:) = (Eepsilon - E) / epsilon;
    end
    deltaInWeights = 1/2 * deltaInWeights;

    e = eye(size(OutWeights, 1));
    deltaOutWeights = zeros(size(OutWeights, 1), 1);
    for j = 1:size(OutWeights, 1)
        OutWeightsEpsilon = OutWeights + repmat(epsilon*e(:,j),1,size(OutWeights,2));
        Eepsilon = meanSquaredError(Data, InWeights, OutWeightsEpsilon, h);
        deltaOutWeights(j,:) = (Eepsilon - E) / epsilon;
    end
    deltaOutWeights = 1/2 * deltaOutWeights;
end