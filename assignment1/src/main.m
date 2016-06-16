%%
% I.2.1
%
clear;
h = figure(1);
randomSeed = rng(43786953);

fprintf('############### I.2.1 ###############\n');
X = [-10:0.01:14];
Y1 = arrayfun(@(x) gauss(x,-1,1), X);
Y2 = arrayfun(@(x) gauss(x,0,2), X);
Y3 = arrayfun(@(x) gauss(x,2,3), X);

plot(X, Y1, 'r-', 'LineWidth', 1.5);
hold on;
plot(X, Y2, 'g-', 'LineWidth', 1.5);
plot(X, Y3, 'b-', 'LineWidth', 1.5);
hold off;
axis ([X(1), X(end), 0, max([Y1 Y2 Y3])]);
legend('[-1, 1]', '[0, 2]', '[2, 3]');
betterPlots(h);
print(h,'-dpng','../figures/I21.png');

%%
% I.2.2
%
fprintf('############### I.2.2 ###############\n');
n = 100;
mu = [1 2]';
Sigma = [0.3 0.2; 0.2 0.2];
R = randn(n,2);
Y1 = resampleGauss(R, mu, Sigma);

plot(Y1(:,1), Y1(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','r');
xlabel('x');
ylabel('y');
betterPlots(h);
print(h,'-dpng','../figures/I22.png');

%%
% I.2.3
%
fprintf('############### I.2.3 ###############\n');
muML = mean(Y1)'
muDist = abs(mu - muML)

hold on;
plot(mu(1), mu(2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','b');
plot(muML(1), muML(2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','g');
hold off;
betterPlots(h);
print(h,'-dpng','../figures/I23.png');

%%
% I.2.4
%
fprintf('############### I.2.4 ###############\n');
SigmaML = zeros(2, 2);
for k=1:n
    t = Y1(k,:)' - muML;
    SigmaML = SigmaML + t * t';
end
SigmaML = SigmaML / n

[EigenVectors, eigenValues] = eig(SigmaML)

e1 = mu + sqrt(eigenValues(1,1)) * EigenVectors(:,1)
e2 = mu + sqrt(eigenValues(2,2)) * EigenVectors(:,2)

plot(Y1(:,1), Y1(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','r');
hold on;
plot([muML(1) e1(1)], [muML(2) e1(2)], 'b-', 'LineWidth', 1.5);
plot([muML(1) e2(1)], [muML(2) e2(2)], 'b-', 'LineWidth', 1.5);
betterPlots(h);
print(h,'-dpng','../figures/I24_1.png');
hold off;

Sigma30 = rotateMatrix(SigmaML, 30);
Sigma60 = rotateMatrix(SigmaML, 60);
Sigma90 = rotateMatrix(SigmaML, 90);

Y2 = resampleGauss(R, mu, Sigma30);
Y3 = resampleGauss(R, mu, Sigma60);
Y4 = resampleGauss(R, mu, Sigma90);

plot(Y2(:,1), Y2(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','r');
hold on;
plot(Y3(:,1), Y3(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','g');
plot(Y4(:,1), Y4(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','b');
legend('30 deg', '60 deg', '90 deg');
betterPlots(h);
print(h,'-dpng','../figures/I24_2.png');
hold off;

zeroAngle = -atand(EigenVectors(1,1)/EigenVectors(2,1))
zeroRotatedY = resampleGauss(R, mu, rotateMatrix(SigmaML, zeroAngle));

plot(Y1(:,1), Y1(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','r');
hold on;
plot(zeroRotatedY(:,1), zeroRotatedY(:,2), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','b');
betterPlots(h);
print(h,'-dpng','../figures/I24_3.png');

%%
% I.4.1
%
clear;
fprintf('############### I.4.1 ###############\n');
trainingData = importdata('../IrisTrain2014.dt');
testData = importdata('../IrisTest2014.dt');

fprintf('Training data:\n');
for k=1:2:5
    C = arrayfun(@(x, y) kNN(trainingData, [x y], k), trainingData(:, 1), trainingData(:, 2));
    acc = 1 - sum(trainingData(:, 3) ~= C)/length(C);
    fprintf('\tk = %d, accuracy = %1.2f\n', k, acc);
end

fprintf('\nTest data:\n');
for k=1:2:5
   C = arrayfun(@(x, y) kNN(trainingData, [x y], k), testData(:, 1), testData(:, 2));
   acc = 1 - sum(testData(:, 3) ~= C)/length(C);
   fprintf('\tk = %d, accuracy = %1.2f\n', k, acc);
end

%%
% I.4.2
%
fprintf('############### I.4.2 ###############\n');
[kBest, kError] = crossValidation(trainingData, 5, [1:2:25])
C = arrayfun(@(x, y) kNN(trainingData, [x y], kBest), testData(:, 1), testData(:, 2));
tError = sum(testData(:, 3) ~= C)/length(C)

%%
% I.4.3
%
fprintf('############### I.4.3 ###############\n');
Means = mean(trainingData(:,1:2), 1)
Vars = var(trainingData(:,1:2), 0, 1)
Stds = std(trainingData(:,1:2), 0, 1);

normTrainingData = fNorm(trainingData, Means, Stds);

normMeans = mean(normTrainingData(:,1:2), 1)
normVars = var(normTrainingData(:,1:2), 0, 1)

normTestData = fNorm(testData, Means, Stds);

normTestMeans = mean(normTestData(:,1:2), 1)
normTestVars = var(normTestData(:,1:2), 0, 1)

[kNormBest, kNormError] = crossValidation(normTrainingData, 5, [1:2:25])
C = arrayfun(@(x, y) kNN(normTrainingData, [x y], kNormBest), normTestData(:, 1), normTestData(:, 2));
tNormError = sum(normTestData(:, 3) ~= C)/length(C)