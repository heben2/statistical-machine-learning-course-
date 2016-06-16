%Works only for a one-layer hidden neurons/layer.
%Large and small learning rate/step must not be equal.
%Use the large step (learning rate) first, and when experience an ascending step,
%stops and use the small step to descent again, until error change is small enough
%to stop, so as to not overfit too much.
%Returns the weights along with the training and test MSEs.
function [InWeights, OutWeights, TrainErrors, TestErrors] = steepestDescent(TrainData, TestData, InWeights, OutWeights, h, hdiff, epsilon, stepLarge, stepSmall)
    deltaError = epsilon + 1;
    oldError = meanSquaredError(TrainData, InWeights, OutWeights, h);
    newError = oldError + 1;
    TrainErrors = [];
    TestErrors = [];
    step = stepLarge;
    
    while (deltaError > epsilon)
        [InWeights, OutWeights, newError] = ...
            takeStep(TrainData, h, hdiff, InWeights, OutWeights, step);

        TrainErrors(1,end+1) = newError;
        TestErrors(1,end+1) = meanSquaredError(TestData, InWeights, OutWeights, h);

        deltaError = oldError - newError;
        %If we went ascended the slope on the other side, take small steps back until another ascend.
        if deltaError <= 0 & step == stepLarge
            [InWeights, OutWeights, newError] = ...
                takeStep(TrainData, h, hdiff, InWeights, OutWeights, step);
            step = stepSmall;
            %we are not done yet, so force deltaError positive.
            deltaError = abs(deltaError);
            TrainErrors(1,end+1) = newError;
            TestErrors(1,end+1) = meanSquaredError(TestData, InWeights, OutWeights, h);
        end
        
        oldError = newError;
    end
    %Do another step if we got an increase in deltaerror in last step 
    %i.e. ascended the slope on the other side. This way we descend to a low 
    if deltaError <= 0
        [InWeights, OutWeights, newError] = ...
            takeStep(TrainData, h, hdiff, InWeights, OutWeights, step);

        TrainErrors(1,end+1) = newError;
        TestErrors(1,end+1) = meanSquaredError(TestData, InWeights, OutWeights, h);
    end
end


function [InWeights, OutWeights, newError] = takeStep(TrainData, h, hdiff, InWeights, OutWeights, step)
    [deltaInWeights, deltaOutWeights] = backPropagation(TrainData, h, hdiff, InWeights, OutWeights);
    deltaInWeights = step*sign(deltaInWeights);
    deltaOutWeights = step*sign(deltaOutWeights);
    InWeights = InWeights - deltaInWeights;
    OutWeights = OutWeights - deltaOutWeights;
    newError = meanSquaredError(TrainData, InWeights, OutWeights, h);
end