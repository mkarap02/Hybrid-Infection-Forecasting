%Function describing the dynamics of SIDAREVH model, that generate predictions.

function [y,dy] = Dynamics(dt, x, buu, bvu, bvv, buv, vaccperday, gamma_i, gamma_a, gamma_d,  gamma_h, ksi_i, ksi_d, mu_a, mu_h)

    dy(1,1) = -buu*x(2,1)*x(1,1) - bvu*x(3,1)*x(1,1) - vaccperday;                         %S : Susceptible
    dy(2,1) = buu*x(2,1)*x(1,1) + bvu*x(3,1)*x(1,1) - ksi_i*x(2,1) - gamma_i*x(2,1);       %I : Infected
    dy(3,1) = bvv*x(2,1)*x(7,1) + buv*x(3,1)*x(7,1) - ksi_d*x(3,1) - gamma_d*x(3,1);       %D : Vaccinated infected
    dy(4,1) = ksi_i*x(2,1) - gamma_a*x(4,1) - mu_a*x(4,1);                                 %A : Hospitalized
    dy(5,1) = gamma_i*x(2,1) + gamma_d*x(3,1) + gamma_a*x(4,1) + gamma_h*x(8,1);           %R : Recovered
    dy(6,1) = mu_a*x(4,1) + mu_h*x(8,1);                                                   %E : Extinct
    dy(7,1) = vaccperday - bvv*x(2,1)*x(7,1) - buv*x(3,1)*x(7,1);                          %V : Vaccinated susceptible 
    dy(8,1) = ksi_d*x(3,1) - gamma_h*x(8,1) - mu_h*x(8,1);                                 %H : Vaccinated hospitalized
    
    y = max(x + dt*dy,0);   %State update

end