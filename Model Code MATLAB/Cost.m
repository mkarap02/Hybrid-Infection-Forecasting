%Function used in model-based infection rates estimatio, for calculating
%the deviations between actual and predicted infected cases.

function [C] = Cost(DataWindow, x, count)

    c2=0; %Unvaccinated infected cost
    c3=0; %Vaccinated infected cost
    
    for i=1:count
        c2 = c2 + (DataWindow.Infected(i)-x(2,i))*((DataWindow.Infected(i)-x(2,i)).');
        c3 = c3 + (DataWindow.VaccinatedInfected(i)-x(3,i))*((DataWindow.VaccinatedInfected(i)-x(3,i)).');
    end

    C = c2 + c3;
    
end