function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%J(theta) = (1/2m)sum from i = 1 to m of (h(theta(x(i)) -y(i))^2
%htheta(x) = ThetaT*x  = Theta0 + Theta1*x1
 

front_coeff = (1/(2*m));

htheta = X*theta ; 
diff = htheta - y ; 
diff2 = diff.^2 ; 
diffsum = sum(diff2);

J = front_coeff.*diffsum



% =========================================================================

end
