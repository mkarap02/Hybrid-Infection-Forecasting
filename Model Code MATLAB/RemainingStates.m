clc;
clear all;

%Import files.
Step1 = readtable('Results_Step1.csv');
days=height(Step1);

CasesData = readtable('CasesData.csv'); %Original Data
data=height(CasesData);

sz = [days 9];
varTypes = ["datetime","double","double","double","double","double","double","double","double"];
varNames = ["Dates","Susceptible","VaccinatedSusceptible","Infected","VaccinatedInfected","Hospitalized","VaccinatedHospitalized","Recovered","Extinct"];
Results = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

%Initializing the values of the table.
for i = 1:days
    Results(i,1)=Step1(i,1);
    Results(i,2)=Step1(i,2);
    Results(i,3)=Step1(i,3);
end

EndDay=CasesData.FirstSampling(data);
enddate=days;

for i=1:data

    InfectionDay=CasesData.FirstSampling(i);
   
    if not(ismissing(CasesData.AdmissionDate(i)))
        HospitalDay=CasesData.AdmissionDate(i);
    end
    
    if (CasesData.PatientState(i)==1) && (not(ismissing(CasesData.AdmissionDate(i))))
        FinalDay=CasesData.DischargedDate(i);
    elseif (CasesData.PatientState(i)==1) && (ismissing(CasesData.AdmissionDate(i)))
        FinalDay=CasesData.RecoveredDate(i);
    elseif (CasesData.PatientState(i)==0) %extinct
        FinalDay=CasesData.DischargedDate(i);
    else
        i
        disp("ERROR IN INITIALIZATION")
        return;
    end

    for k=1:days
        if (Results.Dates(k) == InfectionDay)
            infectiondate=k;
        end
        if not(ismissing(CasesData.AdmissionDate(i)))
            if (Results.Dates(k) == HospitalDay)
                admissiondate=k;
            end
        end        
        if (Results.Dates(k) == FinalDay)
            finaldate=k;
        end
    end


    for j=infectiondate:enddate
        %Susceptibles and Unvaccinated Infections
        if (ismissing(CasesData.VaccineDose1(i)) || (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)<14))
            Results.Susceptible(j)=Results.Susceptible(j)-1;
            Results.Infected(j)=Results.Infected(j)+1;
        end

        %Vaccinated Susceptibles and Vaccinated Infections
        if ((not(ismissing(CasesData.VaccineDose1(i))) ) && (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)>=14))
            Results.VaccinatedSusceptible(j)=Results.VaccinatedSusceptible(j)-1;
            Results.VaccinatedInfected(j)=Results.VaccinatedInfected(j)+1;
        end
    end
    

    if not(ismissing(CasesData.AdmissionDate(i)))
        for j=admissiondate:enddate
            %Unvaccinated Hospitalized
            if (ismissing(CasesData.VaccineDose1(i)) || (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)<14))
                Results.Hospitalized(j)=Results.Hospitalized(j)+1;
                Results.Infected(j)=Results.Infected(j)-1;
            end
            %Vaccinated Hospitalized
            if ((not(ismissing(CasesData.VaccineDose1(i))) ) && (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)>=14))
                Results.VaccinatedHospitalized(j)=Results.VaccinatedHospitalized(j)+1;
                Results.VaccinatedInfected(j)=Results.VaccinatedInfected(j)-1;
            end
        end
    end



    %Recovered and Extinct
    for j=finaldate:enddate
        if (CasesData.PatientState(i)==1) && (ismissing(CasesData.AdmissionDate(i)))
            Results.Recovered(j)=Results.Recovered(j)+1;
            %Unvaccinated Infected
            if (ismissing(CasesData.VaccineDose1(i)) || (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)<14))
                Results.Infected(j)=Results.Infected(j)-1;
            end
            %Vaccinated Infectede
            if ((not(ismissing(CasesData.VaccineDose1(i))) ) && (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)>=14))
                Results.VaccinatedInfected(j)=Results.VaccinatedInfected(j)-1;
            end

        elseif (CasesData.PatientState(i)==1) && (not(ismissing(CasesData.AdmissionDate(i))))
            Results.Recovered(j)=Results.Recovered(j)+1;
            %Unvaccinated Hospitalized
            if (ismissing(CasesData.VaccineDose1(i)) || (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)<14))
                Results.Hospitalized(j)=Results.Hospitalized(j)-1;
            end
            %Vaccinated Hospitalized
            if ((not(ismissing(CasesData.VaccineDose1(i))) ) && (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)>=14))
                Results.VaccinatedHospitalized(j)=Results.VaccinatedHospitalized(j)-1;
            end   
            
        elseif (CasesData.PatientState(i)==0)
            Results.Extinct(j)=Results.Extinct(j)+1;
            %Unvaccinated Hospitalized
            if (ismissing(CasesData.VaccineDose1(i)) || (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)<14))
                Results.Hospitalized(j)=Results.Hospitalized(j)-1;
            end
            %Vaccinated Hospitalized
            if ((not(ismissing(CasesData.VaccineDose1(i))) ) && (CasesData.FirstSampling(i)-CasesData.VaccineDose1(i)>=14))
                Results.VaccinatedHospitalized(j)=Results.VaccinatedHospitalized(j)-1;
            end
        else
            i
            disp("ERROR IN FINAL DATE")
            return;
        end
    end


end
