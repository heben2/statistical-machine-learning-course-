%k_mean takes a m dim matrix of init centroids (also determining k = num rows)
%and the m dim data set (without target variable).
%All init centroids should be unique.
%Ties are broken by chosing the last centroid.
function [clusterCenters, dataClusterSets] = k_mean(init_cents, data)
	k = size(init_cents,1);
	indices = [1:k];
    cents = init_cents;
    tmpCents = init_cents+1;
    centId = [0];
    tmpCentId = [1];
    dataClusterSets = {};

	while ~isequal(cents,tmpCents) | ~isequal(centId,tmpCentId)
		tmpCents = cents;
		tmpCentId = centId;

		D = pdist2(data, cents);
		%a row in D = distances from a datapoint to all centroids (each corresponding to a column)
		DCell = num2cell(D, 2);
		DCellMinDists = cellfun(@(xrow) min(xrow), DCell);
		%Solve ties: chose last centroid - thus we can multiply all min dists with index
		DCell = cellfun(@(xrow, minDist) ...
					arrayfun(@(x, i) (x == minDist)*i, xrow, indices), ...
				DCell, num2cell(DCellMinDists), 'uniformoutput', false);
		%Now select the current centroids for all datapoints
		[skip, centId] = cellfun(@(xrow) max(xrow), DCell, 'uniformoutput', false);
		centId = cell2mat(centId);
		%compute new centroids
		for i = 1:k
			currentIndices = centId;
			currentIndices(currentIndices(:) ~= i, :) = 0;
			%remove all data points not of current cluster i
			dataOfClusteri = diag(currentIndices/i) * data;
			dataOfClusteri(all(dataOfClusteri==0,2),:) = [];
			%update the center and set of cluster i
            cents(i,:) = mean(dataOfClusteri);
            dataClusterSets{i} = dataOfClusteri;
        end
	end
	clusterCenters = cents;
end