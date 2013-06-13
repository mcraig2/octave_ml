% Transform the data from the trained PCA model
function x_trans = pca_predict(model, x, num_components)
  % If the number of dimensions is equal to the size
	% of the model, then we can assume we want to project down
	if (size(x,1) == size(model.U, 1))
		x_trans = model.U(:, 1:num_components)' * x;
	% Otherwise, we are retrieving the data that was
	% previously projected down
	else
		x_trans = x' * model.U(:, 1:num_components)';
	end
end
