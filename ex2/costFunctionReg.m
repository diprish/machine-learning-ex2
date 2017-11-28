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

% ----------------------1. Compute the cost-------------------
n = length(theta);

%hypothesis
h = sigmoid(X * theta);

for i = 1 : m
    % The cost for the ith term before regularization
    J = J - ( y(i) * log(h(i)) )   -  ( (1 - y(i)) * log(1 - h(i)) );

    % Adding regularization term
    for j = 2 : n
        J = J + (lambda / (2*m) ) * ( theta(j) )^2;
    end            
end
J = J/m;

% ----------------------2. Compute the gradients-------------------

j = 1;

for i = 1 : m
    grad(j) = grad(j) + ( h(i) - y(i) ) * X(i,j);
end

for j = 2 : n    
    for i = 1 : m
        grad(j) = grad(j) + ( h(i) - y(i) ) * X(i,j);
    end
    grad(j) = grad(j) + lambda * theta(j); % Change    
end

grad = (1/m) * grad;


% =============================================================

end
