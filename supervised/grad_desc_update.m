% Helper function that converts gradient values into weight update
% values for gradient-descent type of approaches. Essentially, this
% allows for variable learning rates. This function will be expanded
% to include multiple learning rate algorithms, but currently it only
% supports static learning rate (with momentum), and the ADADELTA
% algorithm (Zeiler 2012).
% Input: gradients - cell array, each item is a gradient matrix
%             info - optional, for algorithms that require information
%					 about the previous gradient values, this structure
%					 would be stored here. This structure should also
%				     contain the type of algorithm and the various learning
%					 parameters. Each algorithm requires different parameters.
% Note: this function is used by other algorithms, and doesn't need to
% be called by the user directly, unless the user is creating different
% gradient-descent algorithms.
function [out_grads, info] = grad_desc_update(gradients, info)
	if (!iscell(gradients) || nargin < 2 || isfield(info,'type') == 0)
		fprintf('Error in grad_desc_update...!\n');
		fflush(stdout); return;
	end
	
	% Set up the return value
	out_grads = cell(size(gradients));
	
	% Change algorithm by type
	if (strcmp(info.type,'adadelta') == 1)
		% Set up structure if not already
		if (!isfield(info,'g_sq'))
			info.g_sq = cell(size(gradients));
			for i=1:length(info.g_sq)
				info.g_sq{i} = double(zeros(size(gradients{i}))); end
		end; if(!isfield(info,'x_sq'))
			info.x_sq = cell(size(gradients));
			for i=1:length(info.x_sq)
				info.x_sq{i} = double(zeros(size(gradients{i}))); end
		end
		% Update the weights using the ADADELTA algorithm
		old_grads = info.g_sq;
		for i=1:length(gradients)
			info.g_sq{i} = double((info.decay * info.g_sq{i}) + (gradients{i} .^ 2));
			out_grads{i} = - (double(sqrt(info.x_sq{i} + info.eps)) ./ ...
							  double(sqrt(info.g_sq{i} + info.eps))) .* gradients{i};
			info.x_sq{i} = double((info.decay * info.x_sq{i}) + (out_grads{i} .^ 2));
		end
		% Compute the gradient change
		info.grad_change = mean(mean(double(cell2mat(old_grads)) - double(cell2mat(info.g_sq))));
	elseif (strcmp(info.type,'static') == 1)
		% Set up structure if not already
		if (!isfield(info,'prev_grads'))
			info.prev_grads = cell(size(gradients));
			for i=1:length(info.prev_grads) 
				info.prev_grads{i} = double(zeros(size(gradients{i}))); end
		end
		% Update = learning_rate * gradient + momentum * previous_gradient
		for i=1:length(gradients)
			out_grads{i} = double(info.rate * gradients{i}) + double(info.momentum * info.prev_grads{i});
			out_grads{i} = -out_grads{i}; % For mathematical consistency
		end
		% Compute the gradient change
		info.grad_change = mean(mean(double(cell2mat(info.prev_grads)) - double(cell2mat(out_grads))));
		info.prev_grads = out_grads;
	else
		fprintf('Error in grad_desc_update...confusing type: %s\n',info.type);
		fflush(stdout); return;
	end
end