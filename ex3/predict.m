function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% input layer - a1 = x
% hidden layer - a2 = g(z2)    z2 = theta1a1
% output layer - a3 = g(z3).  z3 = theta2a2

%input layer
X = [ones(m,1) X]; %a1 = x add ones to X matrix

%layer 1
z1 = X*Theta1';
h1 = sigmoid(z1); %returns between 0 and 1

%layer 2
h1 = [ones(m, 1) h1]; %adding ones 

z2 = h1*Theta2';
h2 = sigmoid(z2);

% highest value in earch row and p the position (same as before)

[pval,p] = max(h2,[],2); %vectorised








% =========================================================================


end
