import pandas as pd
import os


def gettingwindows_x(data, columnnum, size_back, size_pred):

    windows_back = []
    win_end = len(data) - (size_back + size_pred)
    for i in range(0, win_end+1):   # from beginning to (window size of days back + prediction size) days before end of data
        a = i+size_back             # window size of days back for every row of final X dataset
        windows_back.append(list(data.iloc[i:a, columnnum]))

    windows_back = pd.DataFrame(windows_back)
    label = data.columns[columnnum]
    windows_back.columns = [label+f"{num}" for num in range(1, (size_back+1))]

    return windows_back


def gettingwindows_y(data, outcomecolnum, size_back, size_pred):

    windows_pred = []
    win_end = len(data)-size_pred
    for i in range(size_back, win_end+1):   # from day where X features stop to (prediction size) days before end of data
        a = i + size_pred                   # window size of days to predict for every row of final Y dataset
        windows_pred.append(list(data.iloc[i:a, outcomecolnum]))

    windows_pred = pd.DataFrame(windows_pred)
    windows_pred.columns = [f"y{num}" for num in range(1, (size_pred + 1))]

    return windows_pred


def windows_table(data, size_back, outcomecolnum, size_pred):

    windowstable = pd.DataFrame()
    for colnum in range(0, len(data.columns)):
        windows_back = gettingwindows_x(data, colnum, size_back, size_pred)
        windowstable = pd.concat([windowstable, windows_back], axis=1)
    windows_pred = gettingwindows_y(data, outcomecolnum, size_back, size_pred)
    windowstable = pd.concat([windowstable, windows_pred], axis=1)

    return windowstable
