function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%grad = zeros(size(initial_theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
hypo = sigmoid(X*theta);
J = -(1/m)*(transpose(y)*log(hypo) + (1 - transpose(y))*log(1 - hypo));

% for i = 1 : size(hypo,1)
%     J = J + (-(1/m)*(y(i)*log(hypo(i)) + (1 - y(i))*log(1 - hypo(i))));
% end

for j = 1:size(grad,1) 
   grad(j) = (1/m) * sum((hypo - y).*X(:,j)); 
end

% for j = 1:size(grad,1)  
%     for i = 1:m
%         grad(j) = grad(j) + (1/m) * ((hypo(i) - y(i))*X(i,j)); 
%     end
% end


% =============================================================

end
