%Script to estimate the four infection rates, by fitting SIDAREVH model to
%the available data of Cyprus.

clc;
clear all;

Data = readtable('Trajectories_01092020_31032022.csv');                    %Import the csv file with the positive cases of Cyprus.
DailyVaccinations = readtable('DailyVaccinations.csv');                    %Import the csv file with the new daily vaccinations.

days=height(Data);  %Number of examined days
n=6; %or n=13
z=days-n;

for i=1:z   %Loop for all days
    
    % 'Dates' function is selecting the examined window (windows of size 7 or 14).
    infmt='dd/MM/yyyy';
    d1=Data.Dates(i); 
    d2=Data.Dates(i+n);
    datetime.setDefaultFormats('defaultdate','dd/MM/yyyy')
    date1=datetime(d1,"InputFormat",infmt);
    date2=datetime(d2,"InputFormat",infmt);
    
    [DataWindow]=Dates(Data,date1,date2);                                  %Table with the data for the examined days
    [VaccinWindow]=Dates(DailyVaccinations,date1,date2);                   %Table with the data for daily vaccinations
    count=height(DataWindow);                                              %Window size

    %Initialization of the infection rates buu, bvu, bvv, buv.
    buu = 0.3899;                                                          %Rate at which unvaccinated people infect other unvaccinated people.
    if (d1<"07/01/2021")                                                   %When vaccinations in Cyprus started.
        bvu = 0;                                                           %Rate at which vaccinated people infect unvaccinated people
        bvv = 0;                                                           %Rate at which vaccinated people infect other vaccinated people
        buv = 0;                                                           %Rate at which unvaccinated people infect vaccinated people
    else
        bvu = 0.19495;       
        bvv = 0.05849;
        buv = 0.11697;
    end
    
    % 'SystemIdentification' function generates predictions and calculates the cost.
    [InitialCost] = SystemIdentification(DataWindow,VaccinWindow,count,buu,bvu,bvv,buv); %Calculation of Initial Cost

    temp=buu; temp1=bvu; temp2=bvv; temp3=buv;
    TempCost=InitialCost;
    flag=0;
    percentage=0.01;
    
    while (flag == 0)
        change(1)=temp+temp*percentage;                                    %Increase of buu
        change(2)=temp-temp*percentage;                                    %Decrease of buu
        change(3)=temp1+temp1*percentage;                                  %Increase of bvu
        change(4)=temp1-temp1*percentage;                                  %Decrease of bvu
        change(5)=temp2+temp2*percentage;                                  %Increase of bvv
        change(6)=temp2-temp2*percentage;                                  %Decrease of bvv
        change(7)=temp3+temp3*percentage;                                  %Increase of buv
        change(8)=temp3-temp3*percentage;                                  %Decrease of buv

        [C(1,1)] = SystemIdentification(DataWindow,VaccinWindow,count,change(1),temp1,temp2,temp3);
        [C(2,1)] = SystemIdentification(DataWindow,VaccinWindow,count,change(2),temp1,temp2,temp3);
        [C(3,1)] = SystemIdentification(DataWindow,VaccinWindow,count,temp,change(3),temp2,temp3);
        [C(4,1)] = SystemIdentification(DataWindow,VaccinWindow,count,temp,change(4),temp2,temp3);
        [C(5,1)] = SystemIdentification(DataWindow,VaccinWindow,count,temp,temp1,change(5),temp3);            
        [C(6,1)] = SystemIdentification(DataWindow,VaccinWindow,count,temp,temp1,change(6),temp3);
        [C(7,1)] = SystemIdentification(DataWindow,VaccinWindow,count,temp,temp1,temp2,change(7));
        [C(8,1)] = SystemIdentification(DataWindow,VaccinWindow,count,temp,temp1,temp2,change(8));
                
        %Find the parameter (minchange) that leads to the minimum cost (min(C)).
        if ( min(C) < TempCost )
            TempCost = min(C);
            minchange = find(C(:,1)==min(C));

            if (minchange==1)
                temp = change(1);
            elseif (minchange==2)
                temp = change(2);
            elseif (minchange==3)
                temp1 = change(3);
            elseif (minchange==4)
                temp1 = change(4);
            elseif (minchange==5)
                temp2 = change(5);
            elseif (minchange==6)
                temp2 = change(6);
            elseif (minchange==7)
                temp3 = change(7);
            elseif (minchange==8)
                temp3 = change(8);
            end

        else 
            flag = 1;
        end  

    end
     
    %Saving optimal infection rates values
    buuC{i,1}=temp;
    bvuC{i,1}=temp1;
    bvvC{i,1}=temp2;
    buvC{i,1}=temp3;
   
end
