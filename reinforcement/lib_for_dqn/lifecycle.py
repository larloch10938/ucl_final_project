import gym
from gym.spaces import Box
import numpy as np
import random


class LifecycleEnv(gym.Env):
    def __init__(self,
                 income_profile,
                 mortality_profile,
                 consumption_shock=0.5,
                 minimum_consumption=0.1,
                 risk_premium=0.05):
        # Set starting variables
        self.starting_wealth = 1
        self.wealth = self.starting_wealth
        self.income = income_profile
        self.mortality = mortality_profile
        self.starting_age = 20
        self.age = self.starting_age
        self.retirement_age = 65
        self.terminal_age = 115
        self.consumption_shock = consumption_shock
        self.minimum_consumption = minimum_consumption
        self.last_consumption = self.minimum_consumption
        # define market returns
        self.risk_free_return = 0.02
        self.equity_return = risk_premium + self.risk_free_return
        self.equity_volatility = 0.25
        self.long_term_consumption_premium = 0.01
        # Observation space: age, from starting age to terminal age and wealth, from 0 to infinite wealth
        self.observation_space = Box(np.array([self.starting_age, 0]),
                                     np.array([self.terminal_age, np.inf]),
                                     shape=(2,))
        # Here we define the choices on consumption and equity allocation as continuos actions
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))

    def step(self, action):
        done = False
        # Calculate consumption and wealth
        action_equity_allocation = (action[0] + 1) / 2
        action_consumption = ((action[1] + 1) / 2)
        # Apply action
        self.risk_return = np.random.normal(
            self.equity_return, self.equity_volatility, 1)[0]
        portfolio_return = ((self.risk_return * action_equity_allocation) +
                            (self.risk_free_return * (1 - action_equity_allocation)))
        # Here income is added only after the portfolio return
        self.wealth = (1 + portfolio_return) * \
            (self.wealth - action_consumption) + self.income[self.age]
        reward = action_consumption * \
            ((1 + self.long_term_consumption_premium) ** (self.age - 20))
        # penalize reward if consumption falls over a certain threshold
        consumption_diff = abs(action_consumption - self.last_consumption)
        if consumption_diff > self.consumption_shock:
            reward = -10
        if action_consumption < self.minimum_consumption:
            reward = -100
            done = True  # do i really need this?
        # terminal conditions: check if agent is broke or dead
        if self.wealth < 0:
            self.wealth = 0
            reward = -100
            done = True
        if random.random() < self.mortality[self.age]:
            done = True
        self.last_consumption = action_consumption
        info = {"age": self.age,
                "wealth": self.wealth,
                "percent_consumption": (action_consumption / self.wealth),
                "consumption": action_consumption,
                "consumption_diff": consumption_diff,
                "equity_allocation": action_equity_allocation,
                "port_return": portfolio_return,
                "reward": reward}
        state = np.array([self.age, self.wealth], dtype=np.float32)
        # Time passes (philosophical consideration here)
        self.age += 1
        return state, reward, done, info

    def render(self, mode='human'):
        # no urgent need for this
        pass

    def reset(self):
        # Resetting age and wealth
        self.wealth = self.starting_wealth
        self.age = self.starting_age
        self.last_consumption = self.minimum_consumption
        state = np.array([self.age, self.wealth], dtype=np.float32)
        return state
