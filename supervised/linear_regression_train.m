% Simple linear regression with parameterize-able basis function
function model = linear_regression_train(x, y, basis_params, basis_type)

if (nargin < 3)
	basis_params = [];
	basis_type = 'linear';
end

% Transform the data with a basis function
x = basis_function(x, basis_type, basis_params);
model.basis_type = basis_type;
model.basis_params = basis_params;

% Add bias to input
x = double([x; ones(1, size(x,2))]);

% Initialize weights
model.w = double(rand(size(x,1), 1));

% If number of features is small (< 1000), we
% can just compute the normal equation. Otherwise,
% stochastic gradient descent
if size(x,1) < 1000
	model.w = pinv(x * x') * x * y';
	model.err = mean((y - (model.w' * x)) .^ 2);
	return;
end

% Stochatic gradient descent algorithm
info.type = 'adadelta';
info.decay = 0.05; info.eps = 1e-8;
max_epochs = 10000;
for i=1:max_epochs
	% Compute the gradient
	model.err = (y - (model.w' * x))';
	grad = {x * model.err};
	% Turn the gradient into a weight update
	[update,info] = grad_desc_update(grad, info);
	% Update the weights
	model.w = model.w - update{1};
	% Print the error
	fprintf('Error: %f\n',mean(err .^ 2)); fflush(stdout);
	% Check stopping condition
	if (mean(model.err .^ 2) <= 1e-3) break; end
	if (strcmp(info.type,'adadelta') == 1)
		if (abs(info.grad_change) < 1e-5) break; end
	elseif (abs(info.grad_change) < 1e-12) break; end
	model.err = mean(model.err);
end

end