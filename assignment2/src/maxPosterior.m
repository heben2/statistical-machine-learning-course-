function mN = maxPosterior(data, alpha, beta)
    Phi = [ones(size(data, 1), 1) data(:, 1:end-1)];
    SN = inv(alpha * eye(size(Phi, 2)) + beta * Phi' * Phi);
    mN = beta * SN * Phi' * data(:, end);
end