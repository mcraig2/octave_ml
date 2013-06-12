% Builds an auto encoder for the given input.
% Assumes that each row is a feature and the
% data is column sampled.
function [model,hidden_outs] = auto_encoder(x, hidden_units, p)

rand('seed', 1);

% Set the inputs/outputs
y = x;

% Normalize the input values
xmean = repmat(mean(x,2), 1, size(x,2));
xstd = repmat(std(x,[],2), 1, size(x,2));
xstd(find(xstd == 0)) = 1;
x = (x - xmean) ./ xstd; % Normalize data

num_inputs = size(x,1);
num_hidden = hidden_units;
num_outputs = size(y,1);

w1 = double((rand(num_inputs+1, num_hidden) * 2) - 1);
w2 = double((rand(num_hidden+1, num_outputs) * 2) - 1);

% Initialize learning parameters
if (nargin < 3)
    decay_p = 0.95;
    eps = 1e-7;
    max_iterations = 20000;
else
    decay_p = p.decay_rate;
    eps = p.eps;
    max_iterations = p.num_iters;
end
egsqg_w1 = 0; egsqg_w2 = 0;
ddx_w1 = 0; ddx_w2 = 0;
exsqx_w1 = 0; exsqx_w2 = 0;

% What activation function?
a_func = 'linear';

% Regularization parameter
lambda = 0.02;

for i=1:max_iterations
    % Forward pass
    hidden = w1' * [x; ones(1,size(x,2))];
    yhat = w2' * [act_func(hidden); ones(1,size(x,2))];

    % Backpropagate the errors
    dEdy = -(y - yhat);
    dydw2 = [act_func(hidden); ones(1,size(x,2))];
    dydh = w2(1:end-1,:);
    dhdv = d_act_func(hidden);
    dvdw1 = [x; ones(1,size(x,2))];
    d_w2 = dydw2 * dEdy' + lambda*w2;
    d_w1 = dvdw1 * ((dydh * dEdy) .* dhdv)' + lambda*w1;

    % Update the weights using ADADELTA algorithm
    egsqg_w1 = (decay_p * egsqg_w1) + (d_w1 .^ 2);
    egsqg_w2 = (decay_p * egsqg_w2) + (d_w2 .^ 2);
    ddx_w1 = (sqrt(exsqx_w1 + eps) ./ sqrt(egsqg_w1 + eps)) .* d_w1;
    ddx_w2 = (sqrt(exsqx_w2 + eps) ./ sqrt(egsqg_w2 + eps)) .* d_w2;
    exsqx_w1 = (decay_p * exsqx_w1) + (ddx_w1 .^ 2);
    exsqx_w2 = (decay_p * exsqx_w2) + (ddx_w2 .^ 2);

    % Update the weights
    w1 = w1 - ddx_w1;
    w2 = w2 - ddx_w2;

    % Forward pass
    hidden = w1' * [x; ones(1, size(x,2))];
    yhat = w2' * [act_func(hidden); ones(1, size(hidden,2))];

    % Save our model
    model.w1 = w1;
    model.w2 = w2;
    hidden_outs = act_func(hidden);

    % Compute goodness of fit
    errs = mean(sqrt(mean((y - yhat) .^ 2, 2)));
    fprintf('RMS: %f Iteration: %d\n', mean(errs), i); fflush(stdout);
end

end

% Logsig function
function v = logsig(in)
    v = (1.0 ./ (1.0 + exp(-in)));
end

% Activation function
function v = act_func(a)
    v = logsig(a);
end

% Derivative of activation function
function dv = d_act_func(v);
    dv = logsig(v) .* (1 - logsig(v));
end
