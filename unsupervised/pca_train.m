% Compute the PCA of the data
function model = pca_train(x)
  path(path, '../preprocessing');

	% Normalize the features
	x = normalize_features(x);
	
	% Compute the PCA
	[U, S, V] = svd((1/size(x,2))*x*x');
	
	% Save the model
	model.U = U;
	model.S = S;
end
