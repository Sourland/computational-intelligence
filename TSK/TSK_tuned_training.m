clc
clear all

%Read Data 
data = readmatrix('train.csv');

%Kanonikopoihsh
for i = 1 : size(data,2)-1
    min_data = min(data(:,i));
    max_data = max(data(:,i));
    new_data(:,i) = (data(:,i)-min_data)/(max_data-min_data); %feature scalling
end
new_data = cat(2,new_data,data(:,end));
new_data = new_data(randperm(size(new_data,1)),:);
num_k=5;

%Assign data
Dtrn = new_data(1:floor(size(new_data,1)*0.6),:);
Dval = new_data(size(Dtrn,1)+1:size(Dtrn,1)+ceil(size(new_data,1)*0.2),:);
Dchk = new_data(size(Dtrn,1)+size(Dval,1)+1:end, :);

%Best feature and radius

features = 15;
radius = 0.2;
%Train Fis
%->Genfis
[indexes,weights] = relieff(Dtrn(:,1:end-1),Dtrn(:,end),10);
genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',radius);
new_fis = genfis(Dtrn(:,indexes(1:features)),Dtrn(:,end),genfis_opt);
%->Anfis
trn_options = anfisOptions('InitialFis',new_fis,'EpochNumber',200);
trn_options.ValidationData = [Dval(:,indexes(1:features)) Dval(:,end)];
[trnFis,trnError,stepSize,valFis,valError] = anfis([Dtrn(:,indexes(1:features)) Dtrn(:,end)],trn_options);

%->Evalfis
y_out = evalfis(valFis, Dchk(:,indexes(1:features))); 
pred_error = Dchk(:,end) - y_out;
    
%Plots
%MF before train
figure;
subplot(2,3,1)
plotmf(new_fis,'input',1);
xlabel('1. Frequency')
        
subplot(2,3,2)
plotmf(new_fis,'input',2);
xlabel('2. Angle of attack')
        
subplot(2,3,3)
plotmf(new_fis,'input',3);
xlabel('3. Chord length')
        
subplot(2,3,4)
plotmf(new_fis,'input',4);
xlabel('4. Free-stream velocity')
        
subplot(2,3,6)
plotmf(new_fis,'input',5);
xlabel('5. Suction side displacement thickness')
suptitle(strcat("Optimal Tsk model MFs before Training"));
name =  strcat('Optimal TSK_model MFs before Training');
saveas(gcf,name,'png');       
figure;

% Learning Curve 
plot([trnError valError], 'LineWidth',2);
xlabel('Number of Iterations');
ylabel('Error');
legend('Training Error', 'Validation Error');
title(strcat("Optimal Tsk model ", strcat(" Learning Curve")));
name = strcat('Optimal TSK_model Learning Curve');
saveas(gcf,name,'png'); 

%MF after train
        
figure;
subplot(2,3,1)
plotmf(valFis,'input',1);
xlabel('1. Frequency')
        
subplot(2,3,2)
plotmf(valFis,'input',2);
xlabel('2. Angle of attack')
        
subplot(2,3,3)
plotmf(valFis,'input',3);
xlabel('3. Chord length')
        
subplot(2,3,4)
plotmf(valFis,'input',4);
xlabel('4. Free-stream velocity')
        
subplot(2,3,6)
plotmf(valFis,'input',5);
xlabel('5. Suction side displacement thickness')
suptitle(strcat("Optimal Tsk model  MFs after Training"));
name =  strcat('Optimal TSK_model MFs after Training');
saveas(gcf,name,'png'); 

%Predictions Plot
figure;
plot([Dchk(:,end) y_out], 'LineWidth',2);
xlabel('input');
ylabel('Values');
legend('Real Value','Prediction Value')
title(strcat("Optimal Tsk model: Prediction versus Real values"));
name = strcat('Optimal TSK_model Model Prediction');
saveas(gcf,name,'png'); 

% Prediction Error
figure;
plot(pred_error, 'LineWidth',2);
xlabel('input');
ylabel('Error');
title(strcat("Optimal Tsk model ", strcat("Prediction Error")));
name = strcat('Optimal TSK_model Prediction Error');
saveas(gcf,name,'png');

%Iterations
    
SSres = sum((Dchk(:,end) - y_out).^2);
SStot = sum((Dchk(:,end) - mean(Dchk(:,end))).^2);
R2 = 1- SSres/SStot;
NMSE = 1-R2;
RMSE = sqrt(mse(y_out,Dchk(:,end)));
NDEI = sqrt(NMSE);

fileID = fopen('Iterations_data_optimal.txt','w');
fprintf(fileID,'TSK_model_optimal');
fprintf(fileID,'\n');
fprintf(fileID,'RMSE = %f\n NMSE = %f\n NDEI = %f\n R2 = %f\n', RMSE, NMSE, NDEI, R2);
fprintf(fileID,'\n');