% Tests the auto encoder
function test_pca

path(path, '../');
path(path, '../preprocessing');

rand('seed', 1);

% Project data from two dimensions into one dimension
% via the PCA algorithm
x = [0:0.01:1; 0:0.01:1];
x = x + normrnd(0, 0.1, size(x));

% Train the PCA model
model = pca_train(x);

% Transform the data
x_t = pca_predict(model, x, 2);
outs = pca_predict(model, x_t, 1);

% Plot the projections
h1 = scatter(x(1,:), x(2,:), 'b'); hold on;
h2 = scatter(outs(1,:), outs(1,:), 'r');

% Draw the lines connecting the un-projected x's with
% the projected x's
for i=1:size(x, 2)
  line([x(1,i); outs(1,i)], [x(2,i); outs(1,i)]);
end
hold off;

end
