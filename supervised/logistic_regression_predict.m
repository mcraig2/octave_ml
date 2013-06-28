% Predicts points with a trained logistic regression model
function y = logistic_regression_predict(x, model)

% Transform the data with a basis function
x = basis_function(x, model.basis_type, model.basis_params);

% Add bias to the points
x = double([x; ones(1, size(x,2))]);

% Compute the predictions
y = logsig(model.w' * x);

end