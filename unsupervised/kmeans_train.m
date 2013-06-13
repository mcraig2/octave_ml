% Given a data matrix X (column sampled), group the
% points into "num_clusters" number of clusters
function model = kmeans_train(x, num_clusters)

% Initialize cluster centers
centroids = x(:,randperm(size(x,2), num_clusters));

% Preallocate variables
distances = zeros(size(centroids, 2), size(x, 2));
old_centroids = centroids;
delta_centroid = zeros(size(centroids));

min_change = 1e-4; change = Inf(1);
while change > min_change
	% Compute the distances from each point to each centroid
	for i=1:num_clusters
		v = x - repmat(centroids(:,i), 1, size(x,2));
		distances(i,:) = arrayfun(@(t) norm(v(:,t)), 1:size(v, 2));
	end
	
	% Find the closest centroid to each point given these distances
	[v,index] = min(distances);
	
	% Now recompute the centroid of the clusters
	for i=1:num_clusters
		centroids(:,i) = mean(x(:, find(index == i)), 2);
	end
	
	% Determine the change in centroids
	delta_centroid = old_centroids - centroids;
	change = mean(arrayfun(@(t) norm(delta_centroid(:,t)), 1:size(delta_centroid,2)));
	old_centroids = centroids;

	%change = min_change - 1;
end

% Store the centroid positions as the model
model.centroids = centroids;

end