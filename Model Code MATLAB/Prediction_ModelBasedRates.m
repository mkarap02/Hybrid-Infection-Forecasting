%Script to predict the infected population using the model-based infection rates.

clc;
clear all;

Data = readtable('Trajectories_01092020_31032022.csv');                    %Import the csv file with the positive cases of Cyprus
ParametersResults = readtable('EstimatedParameters_7daysWindow.csv');      %Import the csv file with the Model-Based estimated infection rates
DailyVaccinations = readtable('DailyVaccinations.csv');                    %Import the csv file with the new daily vaccinations

%Initialization of SIDAREVH model parameters.
gamma_i = 0.0714;                                                          %Recovery rate from infected
gamma_a = 0.0807;                                                          %Recovery rate from hospitalized
gamma_d = 1/14;                                                            %Recovery rate from vaccinated infected 
gamma_h = 1/12.39;                                                         %Recovery rate from vaccinated hospitalized
ksi_i = 0.0053;                                                            %Transition rate from infected to hospitalized
ksi_d = 0.000265;                                                          %Transition rate from vaccinated infected detected to vaccinated acutely symptomatic
mu_a = 0.0085;                                                             %Transition rate from hospitalized to deceased
mu_h = 0.0085;                                                             %Transition rate from vaccinated hospitalized to deceased   

population=920000;                                                         %Total population of Cyprus
days=height(Data);                                                         %Number of examined days
daysofprediction=7;                                                        %Size of window of prediction                                                       
daysofinformation=7;                                                       %Size of the window with the information used for prediction
j = days-( (daysofinformation-1) + daysofprediction);

for i=1:j

    % 'Dates' function is selecting the examined window (windows of size 7 or 14).
    infmt='dd/MM/yyyy';
    d1=Data.Dates(i); 
    d2=Data.Dates(i+(daysofinformation-1));
    
    %Windows with the information data used for prediction
    [DataWindow]=Dates(Data,d1,d2);                                        %Table with the data for the examined days
    [VaccinWindow]=Dates(DailyVaccinations,d1,d2);                         %Table with the data for daily vaccinations
    
    %Windows of prediction (to compare with the real data)
    pd1=Data.Dates(i + daysofprediction);
    pd2=Data.Dates(i + ((daysofinformation-1)+daysofprediction) );
    [PredictionDataWindow]=Dates(Data,pd1,pd2);
    
    buu = ParametersResults.buu(i);                                        %Rate at which unvaccinated people infect other unvaccinated people        
    bvu = ParametersResults.bvu(i);                                        %Rate at which vaccinated people infect unvaccinated people
    bvv = ParametersResults.bvv(i);                                        %Rate at which vaccinated people infect other vaccinated people
    buv = ParametersResults.buv(i);                                        %Rate at which unvaccinated people infect vaccinated people
    
    %Initial conditions:
    x(1,1) = Data.Susceptible(i+ (daysofinformation-1) );                  %S : Susceptible
    x(2,1) = Data.Infected(i+ (daysofinformation-1) );                     %I : Infected detected
    x(3,1) = Data.VaccinatedInfected(i+ (daysofinformation-1) );           %D : Vaccinated infected Detected
    x(4,1) = Data.Hospitalized(i+ (daysofinformation-1) );                 %A : Hospitalized
    x(5,1) = Data.Recovered(i+ (daysofinformation-1) );                    %R : Recovered
    x(6,1) = Data.Extinct(i+ (daysofinformation-1) );                      %E : Extinct
    x(7,1) = Data.VaccinatedSusceptible(i+ (daysofinformation-1) );        %V : Vaccinated Susceptible
    x(8,1) = Data.VaccinatedHospitalized(i+ (daysofinformation-1) );       %H : Vaccinated hospitalized
    
    dt = 1;                   %time increments  
    T = daysofprediction;     %Number of prediction days
    
    % 'Dynamics' function to generate predictions.
    for k=2:T+1
        vaccperday = VaccinWindow.NewPeopleVaccinatedPerDay(k-1);
        x(:,k) = Dynamics(dt, x(:,k-1), buu, bvu, bvv, buv, vaccperday, gamma_i, gamma_a, gamma_d, gamma_h, ksi_i, ksi_d, mu_a, mu_h);
    end
    
    % 'PercentageError' function to calculate the PE for all the windows of prediction.
    [PE] = PercentageError(PredictionDataWindow, x, T);
    PercentageErrorC{i,1} = PE*100;
 
end

MPE = mean(cell2mat(PercentageErrorC));   
STDPE = std(cell2mat(PercentageErrorC));

