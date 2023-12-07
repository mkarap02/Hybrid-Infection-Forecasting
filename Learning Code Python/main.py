# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from class_nn_standard import NNFullyConnected
from class_nn_1dcnn import My1DConv
from matplotlib import pyplot as plt
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import deque

tf.random.set_seed(10)

###########################################################################################
#                                    Auxiliary functions                                  #
###########################################################################################


# Input / Output functions

# def create_file()
def write_to_file(method, data, memory, reps, values, parameters, content):
    filename = 'exps/' + str(method) + '_' + data + '_' + str(memory) + '_' + content + '_' + str(reps) + 'reps.txt'

    with open(filename, mode='a') as L:
        parameters = str(parameters)
        L.write(parameters + " ")
        for item in values:
            item = str(item)
            L.write(item + " ")
        L.write("\n")


def plot_error_sets(file_list, reps):
    fig, ax = plt.subplots()
    for filename in file_list:
        with open(filename, mode='r') as L:
            errors = []
            avg_errors = pd.DataFrame()
            parameters = pd.DataFrame()

            for row in L:
                parameters = pd.concat([parameters, pd.Series(row.split("}")[0])], axis=0)
                row = row.split("}")[1]
                row = row.split(" ")
                row = row[1:]

                row[-1] = row[-1].split("\n")[0]
                errors.append(row)
            errors = pd.DataFrame(errors)
            # parameters = parameters.iloc[::reps, :]
            # parameters = parameters.reset_index(drop=True)

            for r in range(0, len(errors), reps):
                avg_errors = pd.concat([avg_errors, errors.iloc[r:r+reps, :].astype('float').mean(axis=0)], axis=1)
            # print(avg_errors)
            # dates = pd.date_range("15/09/2020", "18/03/2022")
            # dates = pd.DataFrame(dates)
            # preds = pd.concat([dates, avg_errors], axis=1)
            # preds.columns = ['Date', 'beta_predicted']
            # preds.to_csv("exps/beta_predicted.csv", index=False) #15/9/20-18/3/22
        label = filename.rsplit("_")[0].rsplit("/")[1]
        ax.plot(avg_errors, label=label)
        ax.set_xlabel('Days', fontsize=26)
        ax.set_ylabel("Absolute Error", fontsize=26)
        ax.legend(prop={'size': 24})
    # orig = pd.read_csv("data/raw_data.csv")['PositiveCases']
    # ax.plot(orig, label='Original')
    # ax.legend(prop={'size': 24})
    ax.tick_params(axis='both', which='major', labelsize=24)
    fig.suptitle("", size=28)
    plt.show()


def plot_error_errorpercent():
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    avg_errors = pd.DataFrame()

    for filename in ["exps/MLP_raw_pred1_win14_errors_20reps.txt",
                     "exps/MLP_raw_pred1_win14_errors_percent_20reps.txt"]:
        with open(filename, mode='r') as L:
            errors = []
            parameters = pd.DataFrame()

            for row in L:
                parameters = pd.concat([parameters, pd.Series(row.split("}")[0])], axis=0)
                row = row.split("}")[1]
                row = row.split(" ")
                row = row[1:]
                row[-1] = row[-1].split("\n")[0]
                errors.append(row)
            errors = pd.DataFrame(errors)
            # parameters = parameters.iloc[::20, :]
            # parameters = parameters.reset_index(drop=True)

            for r in range(0, len(errors), 20):
                avg_errors = pd.concat([avg_errors, errors.iloc[r:r+20, :].astype('float').mean(axis=0)], axis=1)
        print(avg_errors)

    l1 = ax.plot(avg_errors.iloc[:, 0], label="Absolute Error", color='blue')
    ax.set_xlabel('Days', size=12)
    ax.set_ylabel("Absolute Error", size=12)

    l2 = ax2.plot(avg_errors.iloc[:, 1], label="Absolute Percentage Error", color='red')
    ax2.set_ylabel("Absolute Percentage Error", size=14)

    lns = l1+l2
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs)
    fig.suptitle("Errors (Window 14)", size=18)
    plt.show()


########
# Data #
########


# Load data
def load_data(dataset):

    # beta1, 7th prediction
    if dataset == 'original_beta1_normalized_win7':
        data = pd.read_csv("data/original_beta1_normalized_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta1_normalized_win14':
        data = pd.read_csv("data/original_beta1_normalized_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta1_normalized_win30':
        data = pd.read_csv("data/original_beta1_normalized_win30.csv")
        win, pred = 30, 1

    # beta2, 7th prediction
    elif dataset == 'original_beta2_normalized_win7':
        data = pd.read_csv("data/original_beta2_normalized_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta2_normalized_win14':
        data = pd.read_csv("data/original_beta2_normalized_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta2_normalized_win30':
        data = pd.read_csv("data/original_beta2_normalized_win30.csv")
        win, pred = 30, 1

    # beta3, 7th prediction
    elif dataset == 'original_beta3_normalized_win7':
        data = pd.read_csv("data/original_beta3_normalized_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta3_normalized_win14':
        data = pd.read_csv("data/original_beta3_normalized_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta3_normalized_win30':
        data = pd.read_csv("data/original_beta3_normalized_win30.csv")
        win, pred = 30, 1

    # beta4, 7th prediction
    elif dataset == 'original_beta4_normalized_win7':
        data = pd.read_csv("data/original_beta4_normalized_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta4_normalized_win14':
        data = pd.read_csv("data/original_beta4_normalized_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta4_normalized_win30':
        data = pd.read_csv("data/original_beta4_normalized_win30.csv")
        win, pred = 30, 1

    # interpolated data
    # beta1, 7th prediction
    elif dataset == 'original_beta1_normalized_interpolated_win7':
        data = pd.read_csv("data/original_beta1_normalized_interpolated_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta1_normalized_interpolated_win14':
        data = pd.read_csv("data/original_beta1_normalized_interpolated_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta1_normalized_interpolated_win30':
        data = pd.read_csv("data/original_beta1_normalized_interpolated_win30.csv")
        win, pred = 30, 1

        # beta2, 7th prediction
    elif dataset == 'original_beta2_normalized_interpolated_win7':
        data = pd.read_csv("data/original_beta2_normalized_interpolated_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta2_normalized_interpolated_win14':
        data = pd.read_csv("data/original_beta2_normalized_interpolated_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta2_normalized_interpolated_win30':
        data = pd.read_csv("data/original_beta2_normalized_interpolated_win30.csv")
        win, pred = 30, 1

        # beta3, 7th prediction
    elif dataset == 'original_beta3_normalized_interpolated_win7':
        data = pd.read_csv("data/original_beta3_normalized_interpolated_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta3_normalized_interpolated_win14':
        data = pd.read_csv("data/original_beta3_normalized_interpolated_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta3_normalized_interpolated_win30':
        data = pd.read_csv("data/original_beta3_normalized_interpolated_win30.csv")
        win, pred = 30, 1

        # beta4, 7th prediction
    elif dataset == 'original_beta4_normalized_interpolated_win7':
        data = pd.read_csv("data/original_beta4_normalized_interpolated_win7.csv")
        win, pred = 7, 1
    elif dataset == 'original_beta4_normalized_interpolated_win14':
        data = pd.read_csv("data/original_beta4_normalized_interpolated_win14.csv")
        win, pred = 14, 1
    elif dataset == 'original_beta4_normalized_interpolated_win30':
        data = pd.read_csv("data/original_beta4_normalized_interpolated_win30.csv")
        win, pred = 30, 1

        # 4betas, interpolated, 7th prediction x 4
    elif dataset == 'original_4betas_normalized_interpolated_win7':
        data = pd.read_csv("data/original_4betas_normalized_interpolated_win7.csv")
        win, pred = 7, 4
    elif dataset == 'original_4betas_normalized_interpolated_win14':
        data = pd.read_csv("data/original_4betas_normalized_interpolated_win14.csv")
        win, pred = 14, 4
    elif dataset == 'original_4betas_normalized_interpolated_win30':
        data = pd.read_csv("data/original_4betas_normalized_interpolated_win30.csv")
        win, pred = 30, 4

        # UnvaccInfected, normalized, 7-day prediction
    elif dataset == 'original_UnvaccInfected_normalized_win7':
        data = pd.read_csv("data/original_UnvaccInfected_normalized_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_UnvaccInfected_normalized_win14':
        data = pd.read_csv("data/original_UnvaccInfected_normalized_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_UnvaccInfected_normalized_win30':
        data = pd.read_csv("data/original_UnvaccInfected_normalized_win30.csv")
        win, pred = 30, 7

        # VaccInfected, normalized, 7-day prediction
    elif dataset == 'original_VaccInfected_normalized_win7':
        data = pd.read_csv("data/original_VaccInfected_normalized_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_VaccInfected_normalized_win14':
        data = pd.read_csv("data/original_VaccInfected_normalized_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_VaccInfected_normalized_win30':
        data = pd.read_csv("data/original_VaccInfected_normalized_win30.csv")
        win, pred = 30, 7

        # UnvaccInfected, 7-day prediction
    elif dataset == 'original_UnvaccInfected_win7':
        data = pd.read_csv("data/original_UnvaccInfected_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_UnvaccInfected_win14':
        data = pd.read_csv("data/original_UnvaccInfected_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_UnvaccInfected_win30':
        data = pd.read_csv("data/original_UnvaccInfected_win30.csv")
        win, pred = 30, 7

        # VaccInfected, 7-day prediction
    elif dataset == 'original_VaccInfected_win7':
        data = pd.read_csv("data/original_VaccInfected_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_VaccInfected_win14':
        data = pd.read_csv("data/original_VaccInfected_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_VaccInfected_win30':
        data = pd.read_csv("data/original_VaccInfected_win30.csv")
        win, pred = 30, 7

        # UnvaccInfected, log, 7-day prediction
    elif dataset == 'original_UnvaccInfected_log_win7':
        data = pd.read_csv("data/original_UnvaccInfected_log_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_UnvaccInfected_log_win14':
        data = pd.read_csv("data/original_UnvaccInfected_log_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_UnvaccInfected_log_win30':
        data = pd.read_csv("data/original_UnvaccInfected_log_win30.csv")
        win, pred = 30, 7

        # VaccInfected, log, 7-day prediction
    elif dataset == 'original_VaccInfected_log_win7':
        data = pd.read_csv("data/original_VaccInfected_log_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_VaccInfected_log_win14':
        data = pd.read_csv("data/original_VaccInfected_log_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_VaccInfected_log_win30':
        data = pd.read_csv("data/original_VaccInfected_log_win30.csv")
        win, pred = 30, 7

        # UnvaccInfected + States, log, 7-day prediction
    elif dataset == 'original_UnvaccInfectedStates_log_win7':
        data = pd.read_csv("data/original_UnvaccInfectedStates_log_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_UnvaccInfectedStates_log_win14':
        data = pd.read_csv("data/original_UnvaccInfectedStates_log_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_UnvaccInfectedStates_log_win30':
        data = pd.read_csv("data/original_UnvaccInfectedStates_log_win30.csv")
        win, pred = 30, 7

        # VaccInfected + States, log, 7-day prediction
    elif dataset == 'original_VaccInfectedStates_log_win7':
        data = pd.read_csv("data/original_VaccInfectedStates_log_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_VaccInfectedStates_log_win14':
        data = pd.read_csv("data/original_VaccInfectedStates_log_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_VaccInfectedStates_log_win30':
        data = pd.read_csv("data/original_VaccInfectedStates_log_win30.csv")
        win, pred = 30, 7

        # UnvaccInfected + States, 7-day prediction
    elif dataset == 'original_UnvaccInfectedStates_win7':
        data = pd.read_csv("data/original_UnvaccInfectedStates_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_UnvaccInfectedStates_win14':
        data = pd.read_csv("data/original_UnvaccInfectedStates_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_UnvaccInfectedStates_win30':
        data = pd.read_csv("data/original_UnvaccInfectedStates_win30.csv")
        win, pred = 30, 7

        # VaccInfected + States, 7-day prediction
    elif dataset == 'original_VaccInfectedStates_win7':
        data = pd.read_csv("data/original_VaccInfectedStates_win7.csv")
        win, pred = 7, 7
    elif dataset == 'original_VaccInfectedStates_win14':
        data = pd.read_csv("data/original_VaccInfectedStates_win14.csv")
        win, pred = 14, 7
    elif dataset == 'original_VaccInfectedStates_win30':
        data = pd.read_csv("data/original_VaccInfectedStates_win30.csv")
        win, pred = 30, 7

    else:
        raise Exception('Incorrect dataset entered.')

    return data, win, pred


def denormalize(value, var):
    denormalized = []
    if var == '4betas':
        max1 = pd.read_csv("data/original.csv")['Param'].max()
        max2 = pd.read_csv("data/original.csv")['Param1'].max()
        max3 = pd.read_csv("data/original.csv")['Param2'].max()
        max4 = pd.read_csv("data/original.csv")['Param3'].max()
        for v, m in [(value[0], max1), (value[1], max2), (value[2], max3), (value[3], max4)]:
            denormalized.append(v * m)

    else:
        for v in value:
            if var == 'beta1':
                original = pd.read_csv("data/original.csv")['Param']
            elif var == 'beta2':
                original = pd.read_csv("data/original.csv")['Param1']
            elif var == 'beta3':
                original = pd.read_csv("data/original.csv")['Param2']
            elif var == 'beta4':
                original = pd.read_csv("data/original.csv")['Param3']

            elif var == 'UnvaccInfected':
                original = pd.read_csv("data/original_infected.csv")['UnvaccInfected']  #
            else:
                original = pd.read_csv("data/original_infected.csv")['VaccInfected']  #

            max_value = original.max()
            denormalized.append(v * max_value)
        if len(value) == 1:
            denormalized = denormalized[0]

    return denormalized


def exp(value):  #
    expon = np.exp(value)
    if len(value) == 1:
        expon = expon[0]
    return expon


##########
# Models #
##########

# Create model
def create_model(model, memory_size, layers, rate, epochs, reg, batch, cnn_input, num_filters, kernel_size,
                 pool_size, fc_units, output, order, exog):

    if model == 'MLP':
        params_nn = {'layer_dims': layers, 'learning_rate': rate, 'num_epochs': epochs, 'l2_reg': reg,
                     'minibatch_size': batch}
        f = NNFullyConnected(**params_nn)
        return f, params_nn

    if model == 'ARIMA':
        order = order
        exog = exog
        return order, exog

    if model == '1DCNN':
        params_nn = {'cnn_input_dims': cnn_input, 'kernel_size': kernel_size,
                    'pool_size': pool_size, 'num_filters': num_filters, 'learning_rate': rate, 'num_epochs': epochs,
                     'minibatch_size': batch, 'fc_units': fc_units, 'output': output,  'fc_l2_reg': reg}

        f = My1DConv(**params_nn)
        return f, params_nn

    else:
        raise Exception('Incorrect model entered.')


###########################################################################################
#                                      Main function                                      #
###########################################################################################


def main(params):

    # load data
    data, win, pred = load_data(params['data'])

    # data stats
    n_days, n_features = data.shape

    for r in range(params['n_repeats']):
        errors_per_repeat = []
        predictions_per_repeat = []
        errorspercent_per_repeat = []

        # create memory
        memory = deque(maxlen=params['memory'])

        # create model
        f, params_nn = create_model(model=params['method'], memory_size=params['memory'],
                                    layers=params['layer_dims'], rate=params['learning_rate'],
                                    epochs=params['num_epochs'], reg=params['l2_reg'], batch=params['minibatch_size'],
                                    cnn_input=params['cnn_input'], num_filters=params['filters'],
                                    kernel_size=params['kernel'], pool_size=params['pool'], fc_units=params['fc_units'],
                                    output=params['output'], order=params['order'], exog=params['exog'])

        # observe
        # x = np.array(data.iloc[0, :-params['days_pred']]).reshape(1, n_features-params['days_pred'])
        x = np.array(data.iloc[0, :-4]).reshape(1, n_features-4)  # for 4betas


        # predict
        y_pred = f.prediction(x)[0]  # for y_pred 15
        if 'fft' in params['data']:
            predictions_per_repeat.append(y_pred[0])
        else:
            y_pred_denorm = denormalize(y_pred, params['data'].rsplit('_')[1])
            predictions_per_repeat.append(y_pred_denorm)

        # for forecasting, when params['days_pred']>1
        for d in range(1, params['days_pred']):
            # observe example (x15 or d2-15)
            # x_next = np.array(data.iloc[d, :-params['days_pred']]).reshape(1, n_features - params['days_pred'])
            x_next = np.array(data.iloc[d, :-4]).reshape(1, n_features - 4)  # for 4betas

            # predict next day (y_pred16)
            y_pred = f.prediction(x_next)[0]
            if 'fft' in params['data']:
                predictions_per_repeat.append(y_pred[0])
            else:
                y_pred_denorm = denormalize(y_pred, params['data'].rsplit('_')[1])
                predictions_per_repeat.append(y_pred_denorm)

        for d in range(params['days_pred'], n_days):
            # y = np.array(data.iloc[d - params['days_pred'], -params['days_pred']:]).reshape(1, params['days_pred'])
            y_pred_forerror = predictions_per_repeat[-params['days_pred']]
            y = np.array(data.iloc[d - 1, -4:])  # for 4betas (keep params[days_pred]=1, y_pred_forerror)

            # observe example (x15 or d2-15)
            # x_next = np.array(data.iloc[d, :-params['days_pred']]).reshape(1, n_features-params['days_pred'])
            x_next = np.array(data.iloc[d, :-4]).reshape(1, n_features-4)  # for 4betas

            # evaluation and storage
            if 'fft' in params['data']:
                error = np.round(np.mean(np.abs(np.subtract(y_pred_forerror, y))), 2)
                error_percent = np.round(np.mean((np.abs(np.subtract(y_pred_forerror, y) / y)) * 100), 2)
            else:
                y_denorm = denormalize(y, params['data'].rsplit('_')[1])
                error = np.round(np.mean(np.abs(np.subtract(y_pred_forerror, y_denorm))), 2)
                error_percent = np.round(np.mean((np.abs(np.subtract(y_pred_forerror, y_denorm) / y_denorm))*100), 2)

            #  store error & y_pred
            errors_per_repeat.append(error)
            errorspercent_per_repeat.append(error_percent)

            # train
            xy = np.array(data.iloc[(d-params['days_pred']), :]).reshape(1, n_features)  # (d1-d14, y15)
            memory.append(xy[0])
            memory_df = []
            for example in memory:
                memory_df.append(example)
            memory_df = pd.DataFrame(memory_df)

            # f.train(memory_df.iloc[:, :-params['days_pred']], memory_df.iloc[:, -params['days_pred']:], verbose=1)
            f.train(memory_df.iloc[:, :-4], memory_df.iloc[:, -4:], verbose=1)  # for 4betas

            # predict next day (y_pred16)
            y_pred = f.prediction(x_next)[0]
            if 'fft' in params['data']:
                predictions_per_repeat.append(y_pred[0])
            else:
                y_pred_denorm = denormalize(y_pred, params['data'].rsplit('_')[1])
                predictions_per_repeat.append(y_pred_denorm)

        write_to_file(params['method'], params['data'], params['memory'], params['n_repeats'], errorspercent_per_repeat,
                      params_nn, 'errors_percent')
        write_to_file(params['method'], params['data'], params['memory'], params['n_repeats'], errors_per_repeat,
                      params_nn, 'errors')
        write_to_file(params['method'],  params['data'], params['memory'], params['n_repeats'], predictions_per_repeat,
                      params_nn, 'predictions')


def main_arima(params):

    # load data
    data, win, pred = load_data(params['data'])

    # data stats
    n_days, n_features = data.shape
    # inputdata = n_features - params['days_pred']  # not used

    for r in range(params['n_repeats']):
        errors_per_repeat = []
        predictions_per_repeat = []
        errorspercent_per_repeat = []

        # observe
        x = np.array(data.iloc[0, :-params['days_pred']])

        # create model
        order, exog = create_model(model=params['method'], memory_size=params['memory'],
                             layers=params['layer_dims'], rate=params['learning_rate'],
                             epochs=params['num_epochs'], reg=params['l2_reg'], batch=params['minibatch_size'],
                             cnn_input=params['cnn_input'], num_filters=params['filters'],
                             kernel_size=params['kernel'], pool_size=params['pool'], fc_units=params['fc_units'],
                             output=params['output'], order=params['order'], exog=params['exog'])
        model = SARIMAX(endog=x, exog=exog, order=order, enforce_stationarity=False, initialization='approximate_diffuse')

        # train
        model_fit = model.fit(x, disp=0)

        # predict
        y_pred = model_fit.forecast(params['days_pred'])
        if 'log' in params['data']:
            y_pred_exp = exp(y_pred)
            predictions_per_repeat.append(np.round(y_pred_exp))
        else:
            predictions_per_repeat.append(np.round(y_pred))

        for d in range(1, params['days_pred']):
            # get ground truth (y15)
            # y = np.array(data.iloc[d - 1, -1]).reshape(1, )

            # observe example (x15 or d2-15)
            x_next = np.array(data.iloc[d, :-params['days_pred']])

            model = SARIMAX(endog=x_next, exog=exog, order=order, enforce_stationarity=False, initialization='approximate_diffuse')
            model_fit = model.fit(x_next, disp=0)

            # predict next day (y_pred16)
            y_pred = model_fit.forecast(params['days_pred'])
            if 'log' in params['data']:
                y_pred_exp = exp(y_pred)
                predictions_per_repeat.append(np.round(y_pred_exp))
            else:
                predictions_per_repeat.append(np.round(y_pred))

        for d in range(params['days_pred'], n_days):
            # get ground truth (y15)
            y = np.array(data.iloc[d - params['days_pred'], -params['days_pred']:])
            y_pred_forerror = predictions_per_repeat[-params['days_pred']]

            # observe example (x15 or d2-15)
            x_next = np.array(data.iloc[d, :-params['days_pred']])

            # evaluation and storage
            if 'log' in params['data']:
                y_exp = exp(y)
                error = np.round(np.mean(np.abs(np.subtract(y_pred_forerror, y_exp))), 2)
                error_percent = np.round(np.mean((np.abs(np.subtract(y_pred_forerror, y_exp) / y_exp)) * 100), 2)
            else:
                error = np.round(np.mean(np.abs(np.subtract(y_pred_forerror, y))), 2)
                error_percent = np.round(np.mean((np.abs(np.subtract(y_pred_forerror, y) / y)) * 100), 2)

            #  store error & y_pred
            errors_per_repeat.append(error)
            errorspercent_per_repeat.append(error_percent)

            # train
            model = SARIMAX(endog=x_next, exog=exog, order=order, enforce_stationarity=False, initialization='approximate_diffuse')
            model_fit = model.fit(x_next, disp=0)

            # predict next day
            y_pred = model_fit.forecast(params['days_pred'])
            if 'log' in params['data']:
                y_pred_exp = exp(y_pred)
                predictions_per_repeat.append(np.round(y_pred_exp))
            else:
                predictions_per_repeat.append(np.round(y_pred))
            print(y_pred)

        write_to_file(params['method'], params['data'], params['memory'], params['n_repeats'], errorspercent_per_repeat,
                      "{" + str(order) + "}", 'errors_percent')
        write_to_file(params['method'], params['data'], params['memory'], params['n_repeats'], errors_per_repeat, "{" +
                      str(order) + "}", 'errors')
        write_to_file(params['method'], params['data'], params['memory'], params['n_repeats'], predictions_per_repeat,
                      "{" + str(order) + "}", 'predictions')
