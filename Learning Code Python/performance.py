import pandas as pd
from matplotlib import pyplot as plt
import ast


def perf(method, data, memory, reps, content):
    filename = 'exps/' + str(method) + '_' + data + '_' + str(memory) + '_' + content + '_' + str(reps) + 'reps.txt'
    with open(filename, mode='r') as L:
        results = []
        avg_results = pd.DataFrame()
        parameters = pd.DataFrame()
        for row in L:
            parameters = pd.concat([parameters, pd.Series(row.split("}")[0])], axis=0)
            if "4betas" in filename:
                row = row.split("} ")[1]
                row = row.replace('] [', '], [')
                row = ast.literal_eval(row)
                row = list(row)
            else:
                row = row.split("} ")[1]
                row = row.split(" ")
                row[-1] = row[-1].split("\n")[0]
            results.append(row)
        results = pd.DataFrame(results).iloc[:, :-1]
        #print(results)

        parameters = parameters.iloc[::reps, :]
        parameters = parameters.reset_index(drop=True)

        # Generate raw predictions (when 'predictions' arg is passed)
        count = 0
        for r in range(0, len(results), reps):
            # if "4betas" in filename:
            #     if count == 123:  # col number that gave lower error
            #         raw = results.iloc[r:r + reps, :]
            #     count = count + 1
            # else:
            avg_results = pd.concat([avg_results, results.iloc[r:r+reps, :].astype('float').mean(axis=0)], axis=1)
            # if count == 123:  # col number that gave lower error
            #     raw = results.iloc[r:r + reps, :]
            # count = count+1

        avg_results.columns = range(avg_results.shape[1])

        # Uncomment to generate file with parameters and overall mean, std
        # avg_results_params = pd.concat([parameters, avg_results.transpose().mean(axis=1)], axis=1)
        # avg_results_params.to_csv('exps/4betas_parameters_mape_win30.csv', index=False)

        # for col in avg_results.columns:
        #     print(avg_results[col].mean())
        #     # print(avg_results[col].std())
        #     print(col)
        # plt.plot(avg_results)
        # plt.show()

        # Print mean and std for specific columns of averaged results (when 'errors'/'errors_percent' arg is passed)
        print(parameters.iloc[26, 0])
        print(avg_results.iloc[:, 26].mean())
        print(avg_results.iloc[:, 26].std())

        # Uncomment to save raw predictions
        dates = pd.date_range(start="20/09/2020", end="24/03/2022")  # win7: (start="13/09/2020", end="24/03/2022"),
                                                                     # win14: (start="20/09/2020", end="24/03/2022")
                                                                     # win30: (start="06/10/2020", end="24/03/2022")
        # dates = pd.DataFrame(dates)
        # preds = raw.transpose()  #
        # preds = pd.concat([dates, preds], axis=1)
        # print(preds)
        # preds.to_csv("exps/4betas_predicted_win14_pred7th_mem1_10reps_col123_raw.csv", index=False)


# perf('ARIMA', 'preprocessed_betas_win7_pred7th', 1, 1, 'predictions')
perf('MLP', 'original_beta4_normalized_interpolated_win14', 1, 10, 'errors_percent')
# perf('1DCNN', 'original_beta2_normalized_interpolated_win30', 1, 10, 'errors_percent')
# perf('MLP', 'original_UnvaccInfected_normalized_win30', 30, 10, 'errors_percent')
# perf('ARIMA', 'original_VaccInfected_win30', 1, 1, 'errors_percent')
