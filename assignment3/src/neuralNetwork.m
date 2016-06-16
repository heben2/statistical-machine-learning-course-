% The in weigths are structured such that each row correspond to a the weigths 
% of a  hidden node. Thus the row length should correspond to the dimensionality
% of the data (plus one as the first dimension is the bias) and the number of 
% rows describe the number of hidden nodes. The out weights are
% structured such that every element of the vector correspond two the weight of
% a hidden node, again note that the first element correspond to the bias.
function y = neuralNetwork(h, x, InWeights, OutWeights)
    noOfNodes = size(InWeights, 1);
    x = [1 x]; % Adding the dummy point for the bias w0
    a = arrayfun(@(k) sum(InWeights(k,:).*x), [1:noOfNodes]);
    z = arrayfun(@(ai) h(ai), a);
    y = sum([1 z] .* OutWeights');
end