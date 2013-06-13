% Given cluster centers that were precomputed using
% the k-means algorithm, this classifies new data points
function clusters = kmeans_predict(model, x)

% Compute the distance from each point to the centroid positions
for i=1:size(model.centroids, 2)
	v = x - repmat(model.centroids(:,i), 1, size(x,2));
	distances(i,:) = arrayfun(@(t) norm(v(:,t)), 1:size(v, 2));
end

% Find the closest centroid to each point
[v,clusters] = min(distances);

end