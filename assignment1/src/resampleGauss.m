function SData = resampleGauss(RData, mu, Sigma)
    L = chol(Sigma, 'lower');
    SData = zeros(size(RData, 1),2);
    for k=1:size(RData, 1)
        SData(k,:) = mu + L * RData(k,:)';
    end
end