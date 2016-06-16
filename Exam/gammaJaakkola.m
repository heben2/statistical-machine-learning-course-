%Calculates gamma based on Jaakkola's heuristic.
function y = gammaJaakkola(Data)
	%split data into two sets according to label
	A = arrayfun(@(x) Data(Data(:,end) == x, 1:end-1), unique(Data(:,end)), 'uniformoutput', false);
	%for each data point in first set, pdist with second set and chose minimum.
	A1 = num2cell(A{1}, 2);
	A2 = num2cell(A{2}, 2);
	S1 = cellfun(@(x) min(pdist2(x, A{2})), A1);
	S2 = cellfun(@(x) min(pdist2(x, A{1})), A2);
	sigmaJ = median([S1; S2]);
	y = 1/(2*sigmaJ^2);
end