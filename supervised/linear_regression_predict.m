% Predicts points with a trained linear regression model
function y = linear_regression_predict(x, model)

% Transform the data with a basis function
x = basis_function(x, model.basis_type, model.basis_params);

% Add bias to the points
x = double([x; ones(1, size(x,2))]);

% Compute the predictions
y = model.w' * x;

end