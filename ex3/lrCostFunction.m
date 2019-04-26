function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%
n = length(theta);
xx = X*theta;
h = sigmoid(xx);
funclog = (-y)'*log(h) - (1-y)'*log(1-h);
thetas = theta(2:n,1);
J = ((1/m).*sum(funclog))+(lambda/(2*m)).*sum(thetas.^2);

regulars = theta;
regulars(1)  = 0;

grad = (1/m).*(X'*(h-y)+lambda.*regulars);

end
