# load libraries
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
import investor

def epstein_zin_utility(x, wealth, age, future_utility_function):
    # here I compute the return of the portfolio
    total_return = x[1] * investor.RISK_ASSET_AVERAGE_RETURN + (1 - x[1]) * investor.RISK_FREE_RETURN
    # I compute my expected new wealth given my investment decision and my consumption decision
    new_wealth = investor.INCOME[age] + (wealth - x[0]) * np.exp(total_return)
    # I need to add the bequest motive
    expected_future_utility = (investor.SURVIVAL_PROBABILITY[age] * 
        future_utility_function(new_wealth) ** (1 - investor.GAMMA))
    # lastly I can compute the value function using epstein-zin
    value = ((((1 - investor.DELTA) * x[0] ** (1 - 1 / investor.PSI)) + 
            ((investor.DELTA * expected_future_utility) ** ((1 - 1 / investor.PSI) / 
            (1 - investor.GAMMA)))) ** (1 / (1 - 1 / investor.PSI)))

    return value

def optimize_lifecycle():
    wealth_vector = np.exp(np.linspace(-3, 3, 10))
    consumption_policy = pd.DataFrame(np.zeros((len(investor.AGE_LEVELS), len(wealth_vector))),
        index = investor.AGE_LEVELS, columns = wealth_vector)
    equity_policy = pd.DataFrame(np.zeros((len(investor.AGE_LEVELS), len(wealth_vector))),
        index = investor.AGE_LEVELS, columns = wealth_vector)
    utility_result = pd.DataFrame(np.zeros((len(investor.AGE_LEVELS), len(wealth_vector))),
        index = investor.AGE_LEVELS, columns = wealth_vector)
    
    # Here I solve the problem for the investor at terminal age
    consumption = wealth_vector
    vtplus1 = (((1 - investor.DELTA) * wealth_vector ** (1 - 1 / investor.PSI)) ** 
        (1 / (1 - 1 / investor.PSI)))
    utility = vtplus1
    consumption_policy.loc[investor.END_AGE, :] = consumption
    equity_policy.loc[investor.END_AGE, :] = investor.MIN_EQUITY
    utility_result.loc[investor.END_AGE, :] = utility
    interpolated_utility_policy = PchipInterpolator(wealth_vector, vtplus1)

    # Here I solve the problem for the investor at terminal age minus one
    for wealth in wealth_vector:
        fun = lambda x: -epstein_zin_utility(
            x, wealth, investor.END_AGE - 1, interpolated_utility_policy
            )
        res = minimize(
            fun,
            [wealth / 2, 0.99],
            method = 'SLSQP',
            bounds=[(0, wealth), (investor.MIN_EQUITY, 1)],
            )
        consumption_policy.loc[investor.END_AGE - 1, wealth] = res.x[0]
        equity_policy.loc[investor.END_AGE - 1, wealth] = res.x[1]
        utility_result.loc[investor.END_AGE - 1, wealth] = np.abs(res.fun)

    # I use this loop to reverse-solve the problem until starting age
    for current_age in investor.AGE_LEVELS[::-1][1:]:
        interpolated_utility_policy = PchipInterpolator(
            wealth_vector, utility_result.loc[current_age + 1, :]
            )
        for wealth in wealth_vector:
            fun = lambda x: -epstein_zin_utility(x, wealth, current_age, interpolated_utility_policy)
            res = minimize(
                fun,
                [wealth / 2, 0.99],
                method = 'SLSQP',
                bounds=[(0, wealth), (investor.MIN_EQUITY, 1)],
                )
            consumption_policy.loc[current_age, wealth] = res.x[0]
            equity_policy.loc[current_age, wealth] = res.x[1]
            utility_result.loc[current_age, wealth] = np.abs(res.fun)

    print("Training finished.")
    return (consumption_policy, equity_policy, utility_result)