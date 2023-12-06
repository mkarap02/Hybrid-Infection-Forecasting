function [PE] = PercentageError(PredictionDataWindow, x, T)

    data=0; prediction=0; error=0;

    for m=2:T+1

        data = PredictionDataWindow.Infected(m-1)+PredictionDataWindow.VaccinatedInfected(m-1);
        prediction = x(2,m) + x(3,m);
 
        error = error + abs(prediction/data-1);

    end
    
    PE = error / T;

end