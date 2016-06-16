function y = gauss(x,mu,sigma)
    y = 1 / sqrt(2 * pi * sigma^2) * exp(- 1/(2*sigma^2) * (x - mu)^2);
end