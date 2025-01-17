function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


part1 = -1 * y .* log(sigmoid(X*theta));
part2 = (1 - y) .* log(1 - sigmoid(X*theta));
holder = theta .^ 2;
part3 = (lambda / (2 * m)) * sum(holder(2:size(theta)));  % don't regularize 
                                                          % theta 0

J = sum(part1 - part2) / m;
J = J + part3;

grad(1) = sum((sigmoid(X*theta) - y) .* X(:,1)) / m;
for i = 2:size(theta),
  grad(i) = (sum((sigmoid(X*theta) - y) .* X(:,i)) / m) + (lambda/m) * theta(i);
endfor



% =============================================================

end
