function class = getClass(getDeltas, p)
    deltas = getDeltas(p);
    [lortesprog,maxIdx] = max(deltas(:,1));
    class = deltas(maxIdx, 2);
end