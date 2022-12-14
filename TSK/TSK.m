%% ANFIS EXAMPLE
format compact
clear
clc
close all
addpath('plots');
%% Load data - Split data
data=load('airfoil_self_noise.dat');
[rows, cols] = size(data);
number_of_features = cols - 1;
preproc=1;
[trainingData,evaluationData,testData]=split_scale(data,preproc);
metrics=zeros(4,4);

%% Evaluation function
R_squared = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%% Fuzzy Inference System (FIS) with grid partition
TSK_models = [genfis1(trainingData,2,'gbellmf','constant')...  
              genfis1(trainingData,3,'gbellmf','constant')... 
              genfis1(trainingData,2,'gbellmf','linear')...   
              genfis1(trainingData,3,'gbellmf','linear')];    

for n = 1:length(TSK_models)
% Membership functions plots
 [trainingFis,trainingError,~,evaluationFis,evaluationError] = anfis(trainingData,TSK_models(n),...
     [100 0 0.01 0.9 1.1],[],evaluationData);
    
    % Membership functions plots
    for k = 1:5 % 
        figure();
        plotmf(evaluationFis,'input',k);
        grid on
        title("TSK Model" + n + "Feature "+ k);
        cd('plots')
        saveas(gcf,"Model_" + n + "_Feature" +k+".png")
        cd('..')
    end
    
    % Learning curve plots
    figure();
    plot([trainingError evaluationError]);
    grid on
    xlabel('Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    title("Model " + n + " Learning Curve");
    cd('plots')
    saveas(gcf,"Model_" + n + "_Learning Curve.png")
    cd('..')
    
    % Calculate metrics
    Y = evalfis(testData(:,1:end-1),evaluationFis);
    R2 = R_squared(Y,testData(:,end));
    RMSE = sqrt(mse(Y,testData(:,end)));
    NMSE = 1 - R2; % R2 = 1 - NMSE
    NDEI = sqrt(NMSE);
    metrics(:,n) = [R2; RMSE; NMSE; NDEI];
    
    %Error plot in test data (prediction)
    predict_error = testData(:,end) - Y; 
    figure();
    plot(predict_error);
    grid on;
    xlabel('input');ylabel('Error');
    title("Model " + n + " Prediction Error ");
    cd('plots')
    saveas(gcf, "Model_"+ n + "_prediction_error.png") 
    cd('..')
end

% Results Table
varnames={'Model1', 'Model2', 'Model3', 'Model4'};
rownames={'Rsquared' , 'RMSE' , 'NMSE' , 'NDEI'};
Perf = array2table(metrics,'VariableNames',varnames,'RowNames',rownames)