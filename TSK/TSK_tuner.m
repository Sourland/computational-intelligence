close all; 
clear;
clc

data = csvread('train.csv',1,0);
norm_data = normalize(data(:,1:end-1)); %Normalise data (not the target column)
data = [norm_data(:,1:end) data(:,end)];

% Evaluation function 
R_squared = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%Split the data
training_data = data(1:floor(size(data,1)*0.6),:);
kfold_evaluation_data = data(size(training_data,1)+1:size(training_data,1)+ceil(size(data,1)*0.2),:);
testing_data = data(size(training_data,1)+size(kfold_evaluation_data,1)+1:end, :);

%Set up tuning configurations
total_features = [5 10 15 20 25];
total_radius = [0.2 0.4 0.6 0.8 1];
kfold_choice = 5; %kfold k selection
kfold_data = [training_data; kfold_evaluation_data];
kfold_data_size = length(kfold_data);
error = zeros(length(total_features),length(total_radius));

for i = 1:length(total_features)
    for j = 1:length(total_radius)
        features = total_features(i);
        radius = total_radius(i);
        validation_errors = zeros(kfold_choice);
        
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
        
    end
end