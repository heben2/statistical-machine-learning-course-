%%
% II.1.1
%
clear;fprintf('############### II.1.1 ###############\n');

trainData = importdata('../data/IrisTrain2014.dt');
testData = importdata('../data/IrisTest2014.dt');

getClassFun = lda(trainData);

fprintf('Training data:\n');
C = arrayfun(@(x, y) getClassFun([x y]), trainData(:, 1), trainData(:, 2));
acc = 1 - sum(trainData(:, 3) ~= C)/length(C);
fprintf('\taccuracy = %1.2f\n\n', acc);

fprintf('Test data:\n');
C = arrayfun(@(x, y) getClassFun([x y]), testData(:, 1), testData(:, 2));
acc = 1 - sum(testData(:, 3) ~= C)/length(C);
fprintf('\taccuracy = %1.2f\n\n', acc);

%%
% II.1.2
%
clear;fprintf('############### II.1.2 ###############\n');

trainData = importdata('../data/IrisTrain2014.dt');
testData = importdata('../data/IrisTest2014.dt');

Means = mean(trainData(:,1:2), 1);
Stds = std(trainData(:,1:2), 0, 1);

normTrainData = fNorm(trainData, Means, Stds);
normMeans = mean(normTrainData(:,1:2), 1)
normVars = var(normTrainData(:,1:2), 0, 1)

normTestData = fNorm(testData, Means, Stds);

getClassFun = lda(normTrainData);

fprintf('Training data, normalized:\n');
C = arrayfun(@(x, y) getClassFun([x y]), normTrainData(:, 1), normTrainData(:, 2));
acc = 1 - sum(trainData(:, 3) ~= C)/length(C);
fprintf('\taccuracy = %1.2f\n\n', acc);

fprintf('Test data, normalized:\n');
C = arrayfun(@(x, y) getClassFun([x y]), normTestData(:, 1), normTestData(:, 2));
acc = 1 - sum(testData(:, 3) ~= C)/length(C);
fprintf('\taccuracy = %1.2f\n\n', acc);

%%
% II.2.1
%
clear;fprintf('############### II.2.1 ###############\n');

trainData = importdata('../data/sunspotsTrainStatML.dt');
testData = importdata('../data/sunspotsTestStatML.dt');

w1ML = linearRegression([trainData(:,3:4) trainData(:,6)])
w2ML = linearRegression(trainData(:,5:6))
w3ML = linearRegression(trainData)

y = @(x, wML) [1 x] * wML;

t1 = arrayfun(@(x1, x2) y([x1 x2], w1ML), testData(:,3), testData(:, 4));
t2 = arrayfun(@(x) y(x, w2ML), testData(:,5));
t3 = arrayfun(@(x1, x2, x3, x4, x5) y([x1 x2 x3 x4 x5], w3ML), testData(:,1), ...
              testData(:,2), testData(:,3), testData(:,4), testData(:,5));

t2Line = arrayfun(@(x) y(x, w2ML), [min(testData(:,5)):0.01:max(testData(:,5))]);

h = figure(1);

plot(trainData(:,5), trainData(:,6), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','r');
hold on;
plot(testData(:,5), testData(:,6), 'ok', 'MarkerSize', 6, 'MarkerFaceColor','g');
plot(testData(:,5), t2, 'ok', 'MarkerSize', 6, 'MarkerFaceColor','b');
plot([min(testData(:,5)):0.01:max(testData(:,5))], t2Line, 'b-', 'LineWidth', 1.5);
legend('Training data', 'Test data', 'Linear regression');
ylabel('# of sunspots');
xlabel('Fifth parameter');
betterPlots(h);
print(h, '-depsc2', '../figures/II21_1.eps');
hold off;

rms1 = sqrt(1/size(testData,1)*sum((testData(:,6)-t1).^2))
rms2 = sqrt(1/size(testData,1)*sum((testData(:,6)-t2).^2))
rms3 = sqrt(1/size(testData,1)*sum((testData(:,6)-t3).^2))

plot([1916:1:2011], testData(:, 6), 'k-', 'LineWidth', 1.5);
hold on;
plot([1916:1:2011], t1, 'r-', 'LineWidth', 1.5);
plot([1916:1:2011], t2, 'g-', 'LineWidth', 1.5);
plot([1916:1:2011], t3, 'b-', 'LineWidth', 1.5);
legend('Test data', 'Model 1', 'Model 2', 'Model 3');
ylabel('# of sunspots');
xlabel('Year');
betterPlots(h);
print(h, '-depsc2', '../figures/II21_2.eps');
hold off;

%%
% II.2.2
%
fprintf('############### II.2.2 ###############\n');

% We do not clear, as we need to save these!
rms1ML = rms1;
rms2ML = rms2;
rms3ML = rms3;

trainData = importdata('../data/sunspotsTrainStatML.dt');
testData = importdata('../data/sunspotsTestStatML.dt');

h = figure(1);
y = @(x, wMAP) [1 x] * wMAP;
rms = @(ts) cell2mat(cellfun(@(t) sqrt(1/size(testData,1)*sum((testData(:,6)-t).^2)), ...
                     ts, 'UniformOutput', false));

% Logarithmic scale finding
w1MAP = arrayfun(@(x) maxPosterior([trainData(:,3:4) trainData(:, 6)], x, 1), ...
                 10.^[-10:1:10], 'UniformOutput', false);
w2MAP = arrayfun(@(x) maxPosterior(trainData(:,5:6), x, 1), ...
                 10.^[-10:1:10], 'UniformOutput', false);
w3MAP = arrayfun(@(x) maxPosterior(trainData, x, 1), ...
                 10.^[-10:1:10], 'UniformOutput', false);

t1 = cellfun(@(w) arrayfun(@(x1, x2) y([x1 x2], w), testData(:,3), ...
             testData(:, 4)), w1MAP, 'UniformOutput', false);
rms1 = rms(t1);

t2 = cellfun(@(w) arrayfun(@(x) y(x, w), testData(:,5)), w2MAP, 'UniformOutput', false);
rms2 = rms(t2);

t3 = cellfun(@(w) arrayfun(@(x1, x2, x3, x4, x5) y([x1 x2 x3 x4 x5], w), ...
             testData(:,1), testData(:,2), testData(:,3), testData(:,4), ...
             testData(:,5)), w3MAP, 'UniformOutput', false);
rms3 = rms(t3);

semilogx(10.^[-10:1:10], rms1, 'r-', 'LineWidth', 1.5);
hold on;
semilogx(10.^[-10:1:10], repmat(rms1ML, 1, size(rms1, 2)), 'b--', 'LineWidth', 1.5);
ylabel('RMS');
xlabel('Alpha');
legend('RMS_{MAP}', 'RMS_{ML}');
betterPlots(h);
print(h, '-depsc2', '../figures/II22_1.eps');
hold off;

semilogx(10.^[-10:1:10], rms2, 'r-', 'LineWidth', 1.5);
hold on;
semilogx(10.^[-10:1:10], repmat(rms2ML, 1, size(rms2, 2)), 'b--', 'LineWidth', 1.5);
ylabel('RMS');
xlabel('Alpha');
legend('RMS_{MAP}', 'RMS_{ML}');
betterPlots(h);
print(h, '-depsc2', '../figures/II22_2.eps');
hold off;

semilogx(10.^[-10:1:10], rms3, 'r-', 'LineWidth', 1.5);
hold on;
semilogx(10.^[-10:1:10], repmat(rms3ML, 1, size(rms3, 2)), 'b--', 'LineWidth', 1.5);
ylabel('RMS');
xlabel('Alpha');
legend('RMS_{MAP}', 'RMS_{ML}');
betterPlots(h);
print(h, '-depsc2', '../figures/II22_3.eps');
hold off;


% Scales have been found now for a minima
w1MAP = arrayfun(@(x) maxPosterior([trainData(:,3:4) trainData(:, 6)], x, 1), ...
                 [0:0.001:1], 'UniformOutput', false);
w2MAP = arrayfun(@(x) maxPosterior(trainData(:,5:6), x, 1), ...
                 [0:0.001:1], 'UniformOutput', false);
w3MAP = arrayfun(@(x) maxPosterior(trainData, x, 1), ...
                 [0:1:2000], 'UniformOutput', false);

t1 = cellfun(@(w) arrayfun(@(x1, x2) y([x1 x2], w), testData(:,3), ...
             testData(:, 4)), w1MAP, 'UniformOutput', false);
rms1 = rms(t1);

t2 = cellfun(@(w) arrayfun(@(x) y(x, w), testData(:,5)), w2MAP, 'UniformOutput', false);
rms2 = rms(t2);

t3 = cellfun(@(w) arrayfun(@(x1, x2, x3, x4, x5) y([x1 x2 x3 x4 x5], w), ...
             testData(:,1), testData(:,2), testData(:,3), testData(:,4), ...
             testData(:,5)), w3MAP, 'UniformOutput', false);
rms3 = rms(t3);

plot([0:0.001:1], rms1, 'r-', 'LineWidth', 1.5);
hold on;
plot([0:0.001:1], repmat(rms1ML, 1, size(rms1, 2)), 'b--', 'LineWidth', 1.5);
ylabel('RMS');
xlabel('Alpha');
legend('RMS_{MAP}', 'RMS_{ML}');
betterPlots(h);
print(h, '-depsc2', '../figures/II22_4.eps');
hold off;

plot([0:0.001:1], rms2, 'r-', 'LineWidth', 1.5);
hold on;
plot([0:0.001:1], repmat(rms2ML, 1, size(rms2, 2)), 'b--', 'LineWidth', 1.5);
ylabel('RMS');
xlabel('Alpha');
legend('RMS_{MAP}', 'RMS_{ML}');
betterPlots(h);
print(h, '-depsc2', '../figures/II22_5.eps');
hold off;

plot([0:1:2000], rms3, 'r-', 'LineWidth', 1.5);
hold on;
plot([0:1:2000], repmat(rms3ML, 1, size(rms3, 2)), 'b--', 'LineWidth', 1.5);
ylabel('RMS');
xlabel('Alpha');
legend('RMS_{MAP}', 'RMS_{ML}');
betterPlots(h);
print(h, '-depsc2', '../figures/II22_6.eps');
hold off;