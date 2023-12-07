Classes:
    - class_nn_1dcnn.py is used in main.py
    - class_nn_standard.py is used in main.py

Preprocessing:
    - preprocess_betas.py is used for preprocessing beta data files

Running Scripts:
    - run_script_MLP_win7.py is used for running 80 MLP experiments from random grid combinations of learning parameters
        using original_betaX_normalized_interpolated_win7.csv
    - run_script_MLP_win14.py is used for running 80 MLP experiments from random grid combinations of learning parameters
        using original_betaX_normalized_interpolated_win14.csv
    - run_script_MLP_win30.py is used for running 80 MLP experiments from random grid combinations of learning parameters
        using original_betaX_normalized_interpolated_win30.csv

windows.py contains functions for transforming time series into sliding windows,
    it is used in preprocess_betas.py and preprocess_infected.py

performance.py saves performance (error, percentage error) according to method, data, memory and reps.
main.py is used in all 'run_script' files, contains main algorithms


Order of running files:
    1. preprocess_betas.py / preprocess_infected.py
    2. 'run_script' files
    3. performance.py
