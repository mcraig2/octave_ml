% Performs random hyperparameter search for the given problem.
% Input is a cell array names 'hyper_parms', which holds all the
% possible values each hyperparameters can have, and a function
% handle that trains the function.
function params = hyperparameter_search(x, y, hyper_parms, other_parms, train_fcn, number_runs)
	% Initialize the parameters
	trial_parms = zeros(number_runs, length(hyper_parms));
	for i=1:length(hyper_parms)
		trial_parms(:,i) = hyper_parms{i}(randi(length(hyper_parms{i}), 1, number_runs));
	end
	% Loop through each run and train the model
	errs = zeros(number_runs, 1);
	for i=1:number_runs
		model = train_fcn(x, y, trial_parms(i,:), other_parms);
		errs(i) = model.err;
	end
	% Find the error with the best error
	[es,idx] = min(errs);
	params = trial_parms(idx,:);
end