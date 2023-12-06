from __future__ import division
import pandas as pd
from windows import windows_table
import numpy as np
from matplotlib import pyplot as plt

# original data
betas = pd.read_csv("data/original.csv")
betas = betas.iloc[:, 1:]
betas.columns = ['beta1', 'beta2', 'beta3', 'beta4']


def preprocess(data, pred, threshold, fft):

    data_original = data.copy()

    # normalize (on original data)
    for col in data.columns:
        data[col] = data[col] / data[col].max()
        plt.plot(data[col])
        # plt.show()
    data.to_csv('data/original_normalized.csv', index=False)

    if threshold:
        for b, t in [('beta1', 0.1), ('beta2', 0.8), ('beta3', 0.1), ('beta4', 0.1)]:
            # removing beta < threshold
            for i in range(len(data)):
                if data[b][i] < t:
                    data[b][i] = None

            # linear interpolation
            data[b] = data[b].interpolate(method='linear')

            # first valid position
            i_first_valid = data[b].index.get_loc(data[b].first_valid_index())

            # last valid position
            i_last_valid = data[b].index.get_loc(data[b].last_valid_index())

            # filling in missing values in beginning and end
            for i in range(len(data[b])):
                if i < i_first_valid:
                    data[b][i] = data[b][i_first_valid]
                if i > i_last_valid:
                    data[b][i] = data[b][i_last_valid]
            data[b].to_csv('data/original_' + b + '_normalized_interpolated.csv', index=False)

            # windowing
            data_b = pd.DataFrame(data[b])
            data_7 = windows_table(data_b, 7, 0, pred)
            data_7 = data_7.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_7.to_csv('data/original_' + b + '_normalized_interpolated_win7.csv', index=False)

            data_14 = windows_table(data_b, 14, 0, pred)
            data_14 = data_14.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_14.to_csv('data/original_' + b + '_normalized_interpolated_win14.csv', index=False)

            data_30 = windows_table(data_b, 30, 0, pred)
            data_30 = data_30.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_30.to_csv('data/original_' + b + '_normalized_interpolated_win30.csv', index=False)

    else:
        for b in ['beta1', 'beta2', 'beta3', 'beta4']:
            data_b = pd.DataFrame(data[b])
            data_7 = windows_table(data_b, 7, 0, pred)
            data_7 = data_7.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_7.to_csv('data/original_' + b + '_normalized_win7.csv', index=False)

            data_14 = windows_table(data_b, 14, 0, pred)
            data_14 = data_14.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_14.to_csv('data/original_' + b + '_normalized_win14.csv', index=False)

            data_30 = windows_table(data_b, 30, 0, pred)
            data_30 = data_30.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_30.to_csv('data/original_' + b + '_normalized_win30.csv', index=False)

    if fft:
        # apply fast fourier transform and take absolute values (on original)
        fft = pd.DataFrame()
        for col in data_original.columns:
            f = abs(np.fft.fft(data_original[col]))

            # get the list of frequencies
            freq = [i / len(data_original[col]) for i in list(range(len(data_original[col])))]

            # get the list of spectrums
            spectrum = f.real * f.real + f.imag * f.imag
            nspectrum = spectrum / spectrum[0]

            # plot nspectrum per frequency, with a semilog scale on nspectrum
            plt.semilogy(freq, nspectrum)
            # plt.show()

            nspectrum = pd.DataFrame(nspectrum)

            fft = pd.concat([fft, nspectrum], axis=1)
        fft.columns = ['beta1', 'beta2', 'beta3', 'beta4']
        fft.to_csv('data/orginal_fft.csv', index=False)

        # windows (on fft data)
        for b in ['beta1', 'beta2', 'beta3', 'beta4']:
            fftransformed = pd.DataFrame(fft[b])
            data_7 = windows_table(fftransformed, 7, 0, 7)
            data_7 = data_7.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_7.to_csv('data/original_' + b + '_fft_win7.csv', index=False)

            data_14 = windows_table(fftransformed, 14, 0, 7)
            data_14 = data_14.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_14.to_csv('data/original_' + b + '_fft_win14.csv', index=False)

            data_30 = windows_table(fftransformed, 30, 0, 7)
            data_30 = data_30.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])
            data_30.to_csv('data/original_' + b + '_fft_win30.csv', index=False)


# preprocess(betas, 7, False, False)


def concat_betas(win):
    beta1_x = pd.read_csv('data/original_beta1_normalized_interpolated_win' + str(win) + '.csv').iloc[:, :-1]
    beta2_x = pd.read_csv('data/original_beta2_normalized_interpolated_win' + str(win) + '.csv').iloc[:, :-1]
    beta3_x = pd.read_csv('data/original_beta3_normalized_interpolated_win' + str(win) + '.csv').iloc[:, :-1]
    beta4_x = pd.read_csv('data/original_beta4_normalized_interpolated_win' + str(win) + '.csv').iloc[:, :-1]

    beta1_y = pd.read_csv('data/original_beta1_normalized_interpolated_win' + str(win) + '.csv').iloc[:, -1]
    beta2_y = pd.read_csv('data/original_beta2_normalized_interpolated_win' + str(win) + '.csv').iloc[:, -1]
    beta3_y = pd.read_csv('data/original_beta3_normalized_interpolated_win' + str(win) + '.csv').iloc[:, -1]
    beta4_y = pd.read_csv('data/original_beta4_normalized_interpolated_win' + str(win) + '.csv').iloc[:, -1]

    betas = pd.concat([beta1_x, beta2_x, beta3_x, beta4_x, beta1_y, beta2_y, beta3_y, beta4_y], axis=1)
    betas.to_csv('data/original_4betas_normalized_interpolated_win' + str(win) + '.csv', index=False)


concat_betas(7)
concat_betas(14)
concat_betas(30)


# --------------------------------------------------------------------------------------------------------------------
# PREPROCESSING FOR FILES: preprocessed_betas_win7_pred7th.csv, preprocessed_betas_win14_pred7th.csv,
# preprocessed_betas_win30_pred7th.csv (they all refer to beta1)


def preprocess1(data, window, outcomecol, pred, threshold):  # data is single column

    # windowing
    data = windows_table(data, window, outcomecol, pred)

    # removing beta<0.0001 & replacing with mean of row
    for i in range(len(data)):
        for col in range(len(data.columns)):
            if data.iloc[i, col] < threshold:
                data.iloc[i, col] = None
        data.iloc[i, :] = data.iloc[i, :].fillna(data.iloc[i, :-pred].mean())

    # replacing values still missing with mean of previous row
    for i in range(len(data)):
        for col in range(len(data.columns)):
            if np.isnan(data.iloc[i, col]):
                data.iloc[i, col] = data.iloc[i - 1, :-pred].mean()

    # to keep 7th prediction only
    data = data.drop(columns=["y1", "y2", "y3", "y4", "y5", "y6"])

    return data

# preprocess1(betas, 7, 0, 7, 0.0001)
# preprocess1(betas, 14, 0, 7, 0.0001)
# preprocess1(betas, 30, 0, 7, 0.0001)
