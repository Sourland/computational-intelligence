%TSK TASK 3.2.2
close all; 
clear;
clc

%% Tuning preperation
data = readmatrix('train.csv');
% for i = 1 : size(data,2)-1
%     min_data = min(data(:,i));
%     max_data = max(data(:,i));
%     norm_data(:,i) = (data(:,i)-min_data)/(max_data-min_data); %feature scalling
% end
norm_data = normalize(data(:,1:end-1), 'norm');
shuffle = randperm(size(norm_data,1));
X = norm_data(shuffle,1:end);
Y = data(shuffle,end);
% Evaluation function 
R_squared = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%Split the data
X_train = X(1:floor(size(norm_data,1)*0.6),:);
Y_train = Y(1:floor(size(norm_data,1)*0.6),:);

X_val = X(size(X_train,1)+1:size(X_train,1)+ceil(size(norm_data,1)*0.2),:);
Y_val = Y(size(X_train,1)+1:size(X_train,1)+ceil(size(norm_data,1)*0.2),:);

X_test = X(size(X_train,1)+size(X_val,1)+1:end, :);
Y_test = Y(size(X_train,1)+size(X_val,1)+1:end, :);

%Set up tuning configurations
total_features = [10 15 20];
total_radius = [0.3 0.6 0.9];
kfold_choice = 5; %kfold k selection
X_kfold = [X_train;X_val];
Y_kfold = [Y_train;Y_val];
kfold_data_size = length(X_kfold);
error = zeros(length(total_features),length(total_radius));

best_params = [15 0.2];
min_error = 1e6;
best_indices = 0;
%% Tuning 
tuning = false; % SET TO FALSE TO SKIP 5-FOLD VALIDATION TUNING
if tuning
    for i = 1:length(total_features)
        for j = 1:length(total_radius)
            features = total_features(i);
            radius = total_radius(j);
            validation_errors = zeros(kfold_choice);
            tic
            for k = 1:kfold_choice
                random_idx = randperm(kfold_data_size);

                training_data_idx = random_idx(1:floor(0.8*kfold_data_size));
                temp_X_train = X_kfold(training_data_idx, :);
                temp_Y_train = Y_kfold(training_data_idx, :);
                

                evaluation_data_idx = random_idx(floor(0.8*kfold_data_size)+1:end);
                temp_X_val = X_kfold(evaluation_data_idx, :);
                temp_Y_val = Y_kfold(evaluation_data_idx, :);
                    
                [indices,~] = relieff(temp_X_train,temp_Y_train,10);
                    
                genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',radius);
                new_fis = genfis(temp_X_train(:,indices(1:features)), temp_Y_train,genfis_opt);

                %Training Fis
                training_options = anfisOptions('InitialFis',new_fis,'EpochNumber',100);
                training_options.ValidationData = [temp_X_val(:,indices(1:features)) temp_Y_val];
                
                [training_fis,training_error,stepSize,evaluation_fis,evaluation_error] = anfis([temp_X_train(:,indices(1:features)) temp_Y_tram],training_options);

                %Prediction Error
                validation_error(k) = min(evaluation_error);
            end
                    %Rules
            n_rules(i,j) = size(showrule(evaluation_fis),1);
            %Error

            error(i,j) = sum(validation_error(:)) / kfold_choice;
            if error(i,j) < min_error
                min_error = error(i,j);
                best_params = [features, radius];
                best_indices = indices
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

[indices,~] = relieff(X_train, Y_train, 10);
indices = indices(1:best_features);

X_train = X_train(:,indices);
X_val = X_val(:,indices);

genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', best_radius);
new_fis = genfis(X_train, Y_train, genfis_opt);

%Training Fis
training_options = anfisOptions('InitialFis',new_fis,'EpochNumber',100);
training_options.ValidationData = [X_val Y_val];
[training_fis,training_error,stepSize,evaluation_fis,evaluation_error] =...
    anfis([X_train Y_train], training_options);

Y_pred = evalfis(evaluation_fis, X_test(:,indices)); 
pred_error = Y_test - Y_pred;

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
plot([training_error evaluation_error], 'LineWidth',2);
grid on
xlabel('Number of Iterations');
ylabel('Error');
legend('Training Error', 'Validation Error');
title(strcat("Optimal Tsk model ", strcat(" Learning Curve")));
saveas(gcf,'optimal_learning_curve.png'); 

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
saveas(gcf,'optimal_mf_trained.png'); 

%Predictions Plot
figure;
plot([Y_test Y_pred], 'LineWidth',2);
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
    
SSres = sum((Y_test - Y_pred).^2);
SStot = sum((Y_test - mean(Y_test)).^2);
R2 = 1- SSres/SStot;
NMSE = 1-R2;
RMSE = sqrt(mse(Y_pred,Y_test));
NDEI = sqrt(NMSE);

metrics = [R2 ;NMSE; RMSE; NDEI];
varnames={'Tuned model'};
rownames={'R^2' , 'NMSE' , 'RMSE' , 'NDEI'};
metrics = array2table(metrics,'RowNames',rownames,'VariableNames',varnames);
disp(metrics)
