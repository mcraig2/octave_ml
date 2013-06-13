% Builds an auto encoder for the given input.
% Assumes that each row is a feature and the
% data is column sampled.
function [model,hidden_outs] = auto_encoder(x, hidden_units, p)

path(path, '../preprocessing');

% Set the inputs/outputs
y = x;

% Normalize the input values
x = normalize_features(x);

num_inputs = size(x,1);
num_hidden = hidden_units;
num_outputs = size(y,1);

w1 = double((rand(num_inputs+1, num_hidden) * 2) - 1);
w2 = double((rand(num_hidden+1, num_outputs) * 2) - 1);

% Initialize learning parameters
if (nargin < 3)
    p.decay_rate = 0.05;
    p.eps = 1e-8;
    p.num_iters = 20000;
    p.min_eps = 1e-12;
	p.min_gradient = 1e-5;
end
egsqg_w1 = 0; egsqg_w2 = 0;
ddx_w1 = 0; ddx_w2 = 0;
exsqx_w1 = 0; exsqx_w2 = 0;

% What activation function?
a_func = 'linear';

% Regularization parameter
lambda = 0.02;

% Holds the previous error
prev_errs = Inf(1);

% Holds the original eps value
orig_eps = p.eps;

for i=1:p.num_iters
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
    egsqg_w1 = (p.decay_rate * egsqg_w1) + (d_w1 .^ 2);
    egsqg_w2 = (p.decay_rate * egsqg_w2) + (d_w2 .^ 2);
    ddx_w1 = (sqrt(exsqx_w1 + p.eps) ./ sqrt(egsqg_w1 + p.eps)) .* d_w1;
    ddx_w2 = (sqrt(exsqx_w2 + p.eps) ./ sqrt(egsqg_w2 + p.eps)) .* d_w2;
    exsqx_w1 = (p.decay_rate * exsqx_w1) + (ddx_w1 .^ 2);
    exsqx_w2 = (p.decay_rate * exsqx_w2) + (ddx_w2 .^ 2);

	ddx_w1 = 0.01*d_w1;
	ddx_w2 = 0.01*d_w2;
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
    fprintf('RMS: %f Iteration: %d Eps: %e W1: %e W2: %e\n', mean(errs), i, p.eps, mean(mean(d_w1)), mean(mean(d_w2))); fflush(stdout);
	if (errs > prev_errs)
		p.eps = p.eps * 0.99;
	else
		p.eps = p.eps * (1 + (100*orig_eps)); end
	prev_errs = errs;
	
	% If our eps has fallen below a threshold, we are done training
	grad = max([abs(mean(mean(d_w1))); abs(mean(mean(d_w2)))]);
	if (p.eps < p.min_eps || grad < p.min_gradient)
		break; end
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
