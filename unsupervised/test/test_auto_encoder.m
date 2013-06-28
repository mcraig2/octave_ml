% Tests the auto encoder
function test_auto_encoder

path(path, '../');
path(path, '../preprocessing');

rand('seed', 1);

x = rand(10, 1000);

% With the same number of hidden units as the
% original input, theoretically the network error
% should be 0.
[model,outs] = auto_encoder(x, 10);

% With the number of hidden units less than the
% number of dimensions in the original input, the
% outputs represent the input projected to the
% specified number of dimensions.
x = [0:0.01:1; 0:0.01:1];
x = x + normrnd(0, 0.1, size(x));
[model,outs] = auto_encoder(x, 1);

% Plot the projections
h1 = scatter(x(1,:), x(2,:), 'b'); hold on;
h2 = scatter(outs(1,:), outs(1,:), 'r');

% Draw the lines connecting the un-projected x's with
% the projected x's
x = normalize_features(x);
for i=1:size(x, 2)
  line([x(1,i); outs(1,i)], [x(2,i); outs(1,i)]);
end
hold off;

end
