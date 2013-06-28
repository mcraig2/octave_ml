% Transforms a data matrix by a specified basis function.
% A param array is passed in for each basis function.
function x_new = basis_function(x, basis_type, params)

% Gaussian basis function
if (strcmp(basis_type,'gaussian') == 1)
	x_new = exp(-((x - params(1)) .^ 2) ./ params(2));

% Sigmoidal basis function
elseif (strcmp(basis_type,'sigmoid') == 1)
	x_new = 1.0 ./ (1.0 + exp(-(x - params(1)) ./ params(2)));
	
	
% Polynomial basis function
elseif (strcmp(basis_type,'poly') == 1)
	x_new = x .^ params(1);

% Linear basis function (identity)
elseif (strcmp(basis_type,'linear') == 1)
	x_new = x;

end

end