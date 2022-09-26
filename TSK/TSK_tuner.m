%TSK TASK 3.2.2
close all; 
clear;
clc

%% Tuning preperation
data = readmatrix('train.csv');
norm_data = normalize(data(:,1:end-1)); %Normalise data (not the target column)
data = [norm_data(:,1:end) data(:,end)];

% Evaluation function 
R_squared = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%Split the data
training_data = data(1:floor(size(data,1)*0.6),:);
evaluation_data = data(size(training_data,1)+1:size(training_data,1)+ceil(size(data,1)*0.2),:);
testing_data = data(size(training_data,1)+size(evaluation_data,1)+1:end, :);

%Set up tuning configurations
total_features = [10 15 20 25];
total_radius = [0.2 0.4 0.6 0.8];
kfold_choice = 5; %kfold k selection
kfold_data = [training_data; evaluation_data];
kfold_data_size = length(kfold_data);
error = zeros(length(total_features),length(total_radius));

best_params = [15 0.2];
min_error = 1e6;

%% Tuning 
tuning = false;
if tuning
    for i = 1:length(total_features)
        for j = 1:length(total_radius)
            features = total_features(i);
            radius = total_radius(i);
            validation_errors = zeros(kfold_choice);
            tic
            for k = 1:kfold_choice
                random_idx = randperm(kfold_data_size);

                training_data_idx = random_idx(1:floor(0.8*kfold_data_size));
                kfold_training_data = kfold_data(training_data_idx, :);

                evaluation_data_idx = random_idx(floor(0.8*kfold_data_size)+1:end);
                kfold_evaluation_data = kfold_data(evaluation_data_idx, :);

                [indexes,weights] = relieff(kfold_training_data(:,1:end-1),kfold_training_data(:,end),10);
                    
                genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',radius);
                new_fis = genfis(kfold_training_data(:,indexes(1:features)),kfold_training_data(:,end),genfis_opt);

                %Training Fis
                training_options = anfisOptions('InitialFis',new_fis,'EpochNumber',100);
                training_options.ValidationData = [kfold_evaluation_data(:,indexes(1:features)) kfold_evaluation_data(:,end)];
                [training_fis,training_error,stepSize,evaluation_fis,evaluation_error] = ...
                    anfis([kfold_training_data(:,indexes(1:features)) kfold_training_data(:,end)],training_options);

                %Prediction Error
                validation_error(k) = min(evaluation_error);
            end
                    %Rules
            n_rules(i,j) = size(showrule(evaluation_fis),1);
            %Error

            error(i,j) = sum(validation_error(:)) / kfold_choice;
            if error(i,j) < min_error
                min_error = error(i,j);
                best_params = [i, j];
            end
            toc
        end
    end

    % Plotting Error with Number of Feature and Number of Rules relations
    figure(1)
    subplot(2,2,1);
    plot(total_radius, error(1,:))
    grid on
    title('Number of Feature = 10')
    subplot(2,2,2);
    plot(total_radius, error(2,:))
    grid on
    title('Number of Feature = 15')
    subplot(2,2,3);
    plot(total_radius, error(3,:))
    grid on
    title('Number of Feature = 20')
    subplot(2,2,4);
    plot(total_radius, error(4,:))
    grid on
    title('Number of Feature = 25')
    suptitle('Error - Number of Rules relation');
    saveas(gcf, 'tune_rules.png');

    figure()
    subplot(2,2,1);
    plot(total_features, error(:, 1))
    grid on
    title('Number of Radius = 0.2')
    subplot(2,2,2);
    plot(total_features, error(:, 2))
    grid on
    title('Number of Radius = 0.4')
    subplot(2,2,3);
    plot(total_features, error(:, 3))
    grid on
    title('Number of Radius = 0.6')
    subplot(2,2,4);
    plot(total_features, error(:, 4))
    grid on
    title('Number of Radius = 0.8')
    suptitle('Error - Number of Features relation');
    saveas(gcf, 'tune_features.png');
end

%% Train optimal model
clc
disp("train optimal model")
best_features = best_params(1);
best_radius = best_params(2);

[indexes,weights] = relieff(training_data(:,1:end-1),training_data(:,end),10);

genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', best_radius);
new_fis = genfis(training_data(:,indexes(1:best_features)),training_data(:,end),genfis_opt);

%Training Fis
training_options = anfisOptions('InitialFis',new_fis,'EpochNumber',100);
training_options.ValidationData = [evaluation_data(:,indexes(1:best_features)) evaluation_data(:,end)];
[training_fis,training_error,stepSize,evaluation_fis,evaluation_error] = ...
    anfis([training_data(:,indexes(1:best_features)) training_data(:,end)],training_options);

y_out = evalfis(evaluation_fis, testing_data(:,indexes(1:best_features))); 
pred_error = testing_data(:,end) - y_out;

%% Plots metrics and MS functions
%MF before train
figure;
subplot(2,3,1)
plotmf(new_fis,'input',1);
grid on
xlabel('1. Frequency')
        
subplot(2,3,2)
plotmf(new_fis,'input',2);
grid on
xlabel('2. Angle of attack')
        
subplot(2,3,3)
plotmf(new_fis,'input',3);
grid on
xlabel('3. Chord length')
        
subplot(2,3,4)
plotmf(new_fis,'input',4);
grid on
xlabel('4. Free-stream velocity')
        
subplot(2,3,6)
plotmf(new_fis,'input',5);
grid on
xlabel('5. Suction side displacement thickness')
suptitle(strcat("Optimal Tsk model MFs before Training"));
saveas(gcf,'optimal_mf_no_training.png');       
figure;

% Learning Curve 
plot([trnError valError], 'LineWidth',2);
grid on
xlabel('Number of Iterations');
ylabel('Error');
legend('Training Error', 'Validation Error');
title(strcat("Optimal Tsk model ", strcat(" Learning Curve")));
saveas(gcf,name,'optimal_learning_curve.png'); 

%MF after train
figure;
subplot(2,3,1)
plotmf(evaluation_fis,'input',1);
grid on
xlabel('1. Frequency')
        
subplot(2,3,2)
plotmf(evaluation_fis,'input',2);
grid on
xlabel('2. Angle of attack')
        
subplot(2,3,3)
plotmf(evaluation_fis,'input',3);
grid on
xlabel('3. Chord length')
        
subplot(2,3,4)
plotmf(evaluation_fis,'input',4);
grid on
xlabel('4. Free-stream velocity')
        
subplot(2,3,6)
plotmf(evaluation_fis,'input',5);
grid on
xlabel('5. Suction side displacement thickness')
suptitle(strcat("Optimal Tsk model  MFs after Training"));
saveas(gcf,name,'optimal_mf_trained.png'); 

%Predictions Plot
figure;
plot([testing_data(:,end) y_out], 'LineWidth',2);
grid on
xlabel('input');
ylabel('Values');
legend('Real Value','Prediction Value')
title(strcat("Optimal Tsk model: Prediction versus Real values"));
name = strcat('Optimal TSK_model Model Prediction');
saveas(gcf,name,'png'); 

% Prediction Error
figure;
plot(pred_error, 'LineWidth',2);
grid on
xlabel('input');
ylabel('Error');
title(strcat("Optimal Tsk model ", strcat("Prediction Error")));
name = strcat('Optimal TSK_model Prediction Error');
saveas(gcf,name,'png');

%Model Metrics
    
SSres = sum((testing_data(:,end) - y_out).^2);
SStot = sum((testing_data(:,end) - mean(testing_data(:,end))).^2);
R2 = 1- SSres/SStot;
NMSE = 1-R2;
RMSE = sqrt(mse(y_out,testing_data(:,end)));
NDEI = sqrt(NMSE);

metrics = [R2 NMSE RMSE NDEI];
varnames={'Tuned model'};
rownames={'Rsquared' , 'NMSE' , 'RMSE' , 'NDEI'};
metrics = array2table(metrics,'VariableNames',varnames,'RowNames',rownames);
disp(metrics)
