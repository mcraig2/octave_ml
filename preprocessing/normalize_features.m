% Normalizes the features by removing the mean
% and then dividing by the standard deviation.
% Like always, this assumes x is column sampled.
function x_norm = normalize_features(x)
    % Normalize the features
    xmean = repmat(mean(x,2), 1, size(x,2));
    xstd = std(x,[],2);
    xstd(find(xstd == 0)) = 1;
    xstd = repmat(xstd, 1, size(x,2));
    x_norm = (x - xmean) ./ xstd;
end
