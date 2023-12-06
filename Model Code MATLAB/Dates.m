%Funtion that selects the window of data.

function [Data] = Dates(Data,date1,date2)

    infmt='dd/MM/yyyy';
    datetime.setDefaultFormats('defaultdate','dd/MM/yyyy')

    toDelete = Data.Dates<date1;
    Data(toDelete,:) = [];
   
    toDelete = Data.Dates>date2;
    Data(toDelete,:) = [];
end

