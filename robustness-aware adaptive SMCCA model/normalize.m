function Y = normalize(X, type)

% X in (N, D)
% each row of X corresponds to each subject
% and each column corresponds to each feature

if nargin == 1
    type = 'std';
end

switch type
    case 'center'
        Y = X - mean(X);
    case 'minmax'
        Y = (X - min(X)) ./ (max(X) - min(X) + eps);
    case 'norm'

        num = size(X,1);
        matrix = repmat(mean(X),num,1);

        %X0 = X - mean(X);
        X0 = X - matrix;
        new_matrix = repmat((sqrt(sum(X0 .^ 2)) + eps),num,1);
        
        %Y = X0 ./ (sqrt(sum(X0 .^ 2)) + eps);
        Y = X0 ./ new_matrix;
    case 'std'
        Y = zscore(X);
    otherwise
        error('Error type.');
end
