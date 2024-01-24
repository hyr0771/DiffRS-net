
close all; clear; clc;
%% Load data
%load('ThreeTest.mat');
load('AllBreastData.mat');
for i = 1 : numel(Data_X)
    X{i} = Data_X{i};
end

lambda_1_values = 0.1;
beta_values = 5;
lambda_2_values = 0.5;
lambda_3_values =0.1;



best_params = [];
best_CCC_test = -inf; % 初始化为负无穷

for lambda_1 = lambda_1_values
    for beta = beta_values
        for lambda_2 = lambda_2_values
            for lambda_3 = lambda_3_values
                
                opts.rAdaSMCCA.lambda_1 = lambda_1;
                opts.rAdaSMCCA.beta = beta;
                opts.rAdaSMCCA.lambda_2 = lambda_2;
                opts.rAdaSMCCA.lambda_3 = lambda_3;
                
           

                %% Kfold Cross validation
                n = size(X{1}, 1); 
                k_fold = 5;
                indices = crossvalind('Kfold', n, k_fold);

                fprintf('===================================\n');
                for k = 1 : k_fold
                    fprintf('Current fold: %d\n', k);

                    % Split training data and test data
                    test = (indices == k);
                    train = ~test;
                    for i = 1 : numel(X) 
                    %    testdata = X{1}(train, :)
                    %    tsttdata = X{1}
                        trainData.X{i} = normalize(X{i}(train, :), 'norm');% 200/5=40 40*4=160
                        testData.X{i} = normalize(X{i}(test, :), 'norm');
                    end

                    % Training step
                    % Robustness-aware AdaSMCCA
                    tic;
                    [W.rAdaSMCCA{k}, u.rAdaSMCCA(:, k), v.rAdaSMCCA(:, k), w.rAdaSMCCA(:, k)] = rAdaSMCCA(trainData, opts.rAdaSMCCA);
                    fprintf('Robustness-aware AdaSMCCA: %.3fs\n', toc);
                  

                    % Canonical Correlation Coefficients (CCCs) 
                    % Robustness-aware AdaSMCCA
                    CCC_train.rAdaSMCCA(k, :) = calcCCC(trainData, W.rAdaSMCCA{k});
                    CCC_test.rAdaSMCCA(k, :) = calcCCC(testData, W.rAdaSMCCA{k});
                  
                    if k ~= k_fold
                        fprintf('\n');
                    end
                end
                fprintf('===================================\n');

                %% Weights 
                % Robustness-aware AdaSMCCA
                u.rAdaSMCCA_mean = mean(u.rAdaSMCCA, 2);
                v.rAdaSMCCA_mean = mean(v.rAdaSMCCA, 2);
                w.rAdaSMCCA_mean = mean(w.rAdaSMCCA, 2);
        

                %% CCCs
                % Robustness-aware AdaSMCCA
                CCC_train.rAdaSMCCA_mean = mean(CCC_train.rAdaSMCCA, 1);
                CCC_test.rAdaSMCCA_mean = mean(CCC_test.rAdaSMCCA, 1);
      
                current_CCC_test_mean = mean(CCC_test.rAdaSMCCA_mean);
                
                if current_CCC_test_mean > best_CCC_test
                    best_CCC_test = current_CCC_test_mean;
                    best_params = struct('lambda_1', lambda_1, 'beta', beta, 'lambda_2', lambda_2, 'lambda_3', lambda_3);
                    save('20231231.mat');
                end
            end
        end
    end
end

disp('Best Parameters:');
disp(best_params);
disp(['Best CCC_test Value: ' num2str(best_CCC_test)]);












