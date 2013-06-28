% Tests the k-means clustering algorithm
function test_kmeans

rand('seed', 1);

path(path, '../');

% Set up three clusters of data
x1 = normrnd(0, 3, [2 200]);
x2 = normrnd(10, 3, [2 200]);
x3 = [normrnd(-7.5, 4, [1 200]); normrnd(10, 2, [1 200])];

% Run k-means
x_all = [x1 x2 x3];
model = kmeans_train(x_all, 3);
clusters = kmeans_predict(model, x_all);

% Plot the original clusters
subplot(1,2,1);
scatter(x1(1,:), x1(2,:), 'b'); hold on;
scatter(x2(1,:), x2(2,:), 'r');
scatter(x3(1,:), x3(2,:), 'g'); hold off;
axis square; title('Original Clusters');

% Plot the recovered clusters
subplot(1,2,2);
scatter(x_all(1, find(clusters == 1)), x_all(2, find(clusters == 1)), 'b'); hold on;
scatter(x_all(1, find(clusters == 2)), x_all(2, find(clusters == 2)), 'r');
scatter(x_all(1, find(clusters == 3)), x_all(2, find(clusters == 3)), 'g');
axis square; title('Recovered Clusters with Cluster Centers');

% Plot the cluster centers
scatter(model.centroids(1,1), model.centroids(2,1), 30, 'k');
scatter(model.centroids(1,2), model.centroids(2,2), 30, 'k');
scatter(model.centroids(1,3), model.centroids(2,3), 30, 'k'); hold off;

end