% The logistic sigmoid function
function v = logsig(in)
	v = 1.0 ./ (1.0 + exp(-in));
end