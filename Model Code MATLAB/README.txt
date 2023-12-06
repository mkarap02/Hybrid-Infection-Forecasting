SIDAREVH model's states calculation:
   Order of running files:
	1. Susceptible_VaccinatedSusceptible_States.m
	2. RemainingStates.m

Infection rates model-based estimation:
   Order of running files:
	1. ParametersEstimation.m
		1.1 Dates.m
		1.2 SystemIdentification.m
		1.3 Dynamics.m
		1.4 Cost.m
   (1.1, 1.2, 1.3, 1.4 are functions called by ParametersEstimation.m script)


Prediction of the infected population using the model-based estimated infection rates:
   Order of running files:
	1. Prediction_ModelBasedRates.m
		1.1 Dates.m
		1.2 Dynamics.m
		1.3 PercentageError.m
   (1.1, 1.2, 1.3 are functions called by Prediction_ModelBasedRates.m script)


Prediction of the infected population using the learning-based estimated infection rates:
   Order of running files:
	1. Prediction_LearningBasedRates.m
		1.1 Dates.m
		1.2 Dynamics.m
		1.3 PercentageError.m
   (1.1, 1.2, 1.3 are functions called by Prediction_LearningBasedRates.m script)