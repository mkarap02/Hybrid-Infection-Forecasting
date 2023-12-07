%Function that takes as input the values of the four infection rates,
%generate the predictions, and calculates the cost.

function [C] = SystemIdentification(DataWindow,VaccinWindow,count,buu,bvu,bvv,buv)
  
    %Initial conditions
    x(1,1) = DataWindow.Susceptible(1);                                    %S : Susceptible
    x(2,1) = DataWindow.Infected(1);                                       %I : Infected detected
    x(3,1) = DataWindow.VaccinatedInfected(1);                             %D : Vaccinated infected Detected
    x(4,1) = DataWindow.Hospitalized(1);                                   %A : Hospitalized
    x(5,1) = DataWindow.Recovered(1);                                      %R : Recovered
    x(6,1) = DataWindow.Extinct(1);                                        %E : Extinct
    x(7,1) = DataWindow.VaccinatedSusceptible(1);                          %V : Vaccinated Susceptible
    x(8,1) = DataWindow.VaccinatedHospitalized(1);                         %H : Vaccinated hospitalized
    
    %Initialization of SIDAREVH model parameters
    gamma_i = 0.0714;                                                      %Recovery rate from infected
    gamma_a = 0.0807;                                                      %Recovery rate from hospitalized
    gamma_d = 1/14;                                                        %Recovery rate from vaccinated infected 
    gamma_h = 1/12.39;                                                     %Recovery rate from vaccinated hospitalized
    ksi_i = 0.0053;                                                        %Transition rate from infected to hospitalized
    ksi_d = 0.000265;                                                      %Transition rate from vaccinated infected detected to vaccinated hospitalized
    mu_a = 0.0085;                                                         %Transition rate from hospitalized to deceased
    mu_h = 0.0085;                                                         %Transition rate from vaccinated hospitalized to deceased
        
    dt = 1;                                                                %Time increments  
    T = count/dt;

    %Function that generates the predictions
    for k=2:T
        vaccperday = VaccinWindow.NewPeopleVaccinatedPerDay(k-1);
        x(:,k) = Dynamics(dt, x(:,k-1), buu, bvu, bvv, buv, vaccperday, gamma_i, gamma_a, gamma_d, gamma_h, ksi_i, ksi_d, mu_a, mu_h);
    end

    %Function that calculates Cost
        %x: the predicted data
        %DataWindow: the actual data
    [C] = Cost(DataWindow, x, count);
    
end