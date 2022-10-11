clc
clear
close all

%% Control System params
Kp=1.87
c=-1.11;
Ka=1;
K=10*Ka*Kp;
Ki=Kp*(-c)

%% Transfer Functions
Gc=zpk(c, 0, Kp);
Gp=zpk([],[-1 -9],10);

%Gewmwtrikos topos rizwn
Ac=Gc*Gp;


figure
rlocus(Ac);
hold on
grid on
xlim([-10 0])
%Step
Hc=feedback(Ac,1 ,-1);
figure 
step(Hc)
stepinfo(Hc)
grid on