# Hyperparameters
START_AGE = 20
END_AGE = 115
RETIREMENT_AGE = 70
SURVIVAL_PROBABILITY = [0.99] * END_AGE
INCOME = [1.0] * END_AGE
AGE_LEVELS = list(range(START_AGE, END_AGE))

DELTA = 0.96
PSI = 0.1
GAMMA = 10

MIN_EQUITY = 0.2
initial_income = 1
last_consumption = 1
income_volatility = 0.05
RISK_ASSET_AVERAGE_RETURN = 0.05
risk_asset_volatility = 0.2
RISK_FREE_RETURN = 0.1