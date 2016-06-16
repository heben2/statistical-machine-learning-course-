function [InWeights, OutWeights, TrainErrors, TestErrors] = steepestDescent(TrainData, TestData, InWeights, OutWeights, h, hdiff, epsilon, step)
    deltaError = epsilon + 1;
    oldError = meanSquaredError(TrainData, InWeights, OutWeights, h);
    elderError = oldError + 1;
    newError = elderError + 1;
    errorChange = 1;
    TrainErrors = [];
    TestErrors = [];
    
    while (deltaError > epsilon) & errorChange
        [deltaInWeights, deltaOutWeights] = backPropagation(TrainData, h, hdiff, InWeights, OutWeights);
        deltaInWeights = step*sign(deltaInWeights); 
        deltaOutWeights = step*sign(deltaOutWeights);
        InWeights = InWeights - deltaInWeights;
        OutWeights = OutWeights - deltaOutWeights;
        newError = meanSquaredError(TrainData, InWeights, OutWeights, h);

        TrainErrors(1,end+1) = newError;
        TestErrors(1,end+1) = meanSquaredError(TestData, InWeights, OutWeights, h);

        deltaError = abs(newError - oldError);
        errorChange = elderError ~= newError;
        elderError = oldError;
        oldError = newError;
    end
end