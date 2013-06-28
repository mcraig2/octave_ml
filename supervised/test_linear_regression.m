% Test the linear regression
function test_linear_regression

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Linear case         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
basis_type = 'linear';
x = [-3:0.1:3];
y = x + normrnd(0, 0.8, size(x));
% Train the linear regression model
h_params{1} = [0.01 0.03 0.1 0.3 1 3 10 30];
h_params{2} = [0.01 0.03 0.1 0.3 1 3 10 30];
params = hyperparameter_search(x, y, h_params, basis_type, @linear_regression_train, 30);
model = linear_regression_train(x, y, params, basis_type);
% Predict on the data (usually you'd want a separate test set)
yhat = linear_regression_predict(x, model);
% Plot the results
subplot(2,2,1);
scatter(x,y); hold on;
plot(x,yhat,'-r','LineWidth',2); hold off;
title('Linear Basis Function');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Nonlinear case      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
basis_type = 'gaussian';
x = [-3:0.1:3];
y = sin(x) + normrnd(0, 0.25, size(x));
% Train the linear regression model
params = hyperparameter_search(x, y, h_params, basis_type, @linear_regression_train, 30);
model = linear_regression_train(x, y, params, basis_type);
% Predict the data
yhat = linear_regression_predict(x, model);
% Plot the results
subplot(2,2,2);
scatter(x,y); hold on;
plot(x,yhat,'-r','LineWidth',2); hold off;
title('Gaussian Basis Function');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Polynomial case      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
basis_type = 'poly';
x = [-3:0.1:3];
y = sin(x) + normrnd(0, 0.25, size(x));
% Train the linear regression model
h_params = {[2 3 4 5 6 7 8 9 10]};
params = hyperparameter_search(x, y, h_params, basis_type, @linear_regression_train, 30);
model = linear_regression_train(x, y, params, basis_type);
% Predict the data
yhat = linear_regression_predict(x, model);
% Plot the results
subplot(2,2,3);
scatter(x,y); hold on;
plot(x,yhat,'-r','LineWidth',2); hold off;
title('Polynomial Basis Function');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Sigmoid case       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
basis_type = 'sigmoid';
x = [-3:0.1:3];
y = sin(x) + normrnd(0, 0.25, size(x));
% Train the linear regression model
h_params{1} = [0.01 0.03 0.1 0.3 1 3 10 30];
h_params{2} = [0.01 0.03 0.1 0.3 1 3 10 30];
params = hyperparameter_search(x, y, h_params, basis_type, @linear_regression_train, 30);
model = linear_regression_train(x, y, params, basis_type);
% Predict the data
yhat = linear_regression_predict(x, model);
% Plot the results
subplot(2,2,4);
scatter(x,y); hold on;
plot(x,yhat,'-r','LineWidth',2); hold off;
title('Sigmoid Basis Function');

end