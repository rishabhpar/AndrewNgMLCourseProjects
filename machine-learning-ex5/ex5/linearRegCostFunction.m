function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

part1 = sum((X*theta - y) .^ 2);
holder = theta;
part2 = lambda * sum(holder(2:length(theta)) .^ 2);   %don't regularize the first col
J = (part1 + part2) / (2*m);

holder(1) = 0;
grad = ((1/m) * sum((X*theta - y) .* X)) + (lambda * holder / m);

grad = ((X' * (X*theta - y)) + (lambda*theta)) / m;
grad(1) = sum((X*theta - y) .* X(:,1)) / m;
% =========================================================================

grad = grad(:);

end
