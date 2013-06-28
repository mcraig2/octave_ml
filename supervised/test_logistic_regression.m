% Test the logistic regression
function test_logistic_regression

% Load the datasets
path(path,'../datasets');
lin_d = load('linear_classification.mat');
nonlin_d = load('nonlinear_classification.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Linear case         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,2,1);
basis_type = 'linear';
x = lin_d.X'; y = lin_d.y';
% Train the classifier
model = logistic_regression_train(x, y, [], 'linear');
% Plot the dataset
xtest = [];
meshs = [0:0.1:5];
for i=1:length(meshs)
	for j=1:length(meshs)
		xtest = [xtest [meshs(i); meshs(j)]]; end; end
yhat = logistic_regression_predict(xtest, model);
[meshx,meshy] = meshgrid(meshs, meshs);
yhat = reshape(yhat, length(meshs), length(meshs));
contourf(meshx, meshy, yhat); hold on;
scatter(x(1, find(y == 0)), x(2, find(y == 0)), 30); hold on;
scatter(x(1, find(y == 1)), x(2, find(y == 1)), 30, 'r');
title('Linear Basis Function');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Nonlinear case      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,2,2);
basis_type = 'sigmoid';
h_params = [0.3 10];
% Train the classifier
model = logistic_regression_train(x, y, h_params, 'sigmoid');
% Plot the dataset
xtest = [];
meshs = [0:1:5];
for i=1:length(meshs)
	for j=1:length(meshs)
		xtest = [xtest [meshs(i); meshs(j)]]; end; end
yhat = logistic_regression_predict(xtest, model);
[meshx,meshy] = meshgrid(meshs, meshs);
yhat = reshape(yhat, length(meshs), length(meshs));
contourf(meshx, meshy, yhat); hold on;
scatter(x(1, find(y == 0)), x(2, find(y == 0)), 30); hold on;
scatter(x(1, find(y == 1)), x(2, find(y == 1)), 30, 'r');
title('Gaussian Basis Function');

end