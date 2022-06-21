import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np


class LifecycleEnv(gym.Env):
    def __init__(self):
        # Set starting variables
        self.wealth = np.zeros(1)
        self.starting_income = 100
        self.income = self.starting_income
        self.starting_age = 19
        self.age = self.starting_age
        self.retirement_age = 65
        self.terminal_age = 115
        # Observatioon space: age, from starting age to terminal age and wealth, from 0 to infinite wealth
        self.observation_space = Box(np.array([self.starting_age, 0]), np.array(
            [self.terminal_age, np.inf]), shape=(2,))
        # Here we define the choices on consumption and equity allocation
        self.action_space = Box(np.array([0, 0]), np.array([1, 1]))

    def step(self, action):
        reward = 0
        # Income reduces when you retire
        if self.age >= self.retirement_age:
            self.income = 0
        # define market returns
        risk_return = 0.05
        risk_free_return = 0.01
        # transformation
        action_equity_allocation = action[0]
        action_consumption = action[1] * self.wealth
        # Apply action
        portfolio_return = (risk_return *
                            action_equity_allocation) + (risk_free_return *
                                                         (1 - action_equity_allocation))
        # Here income is added only after the portfolio return
        self.wealth = (1 + portfolio_return) * (self.wealth -
                                                action_consumption) + self.income
        if self.wealth < 0:
            self.wealth = 0
            reward += -1000
            done = True
        reward += action_consumption
        # Complete step if agent is older than terminal age
        if self.age == self.terminal_age:
            done = True
            reward = 0
        else:
            done = False
        # Time passes (philosophical consideration here)
        self.age += 1
        # Placeholder for info
        info = {"age": self.age, "wealth": self.wealth,
                "consumption": action_consumption, 
                "equity_allocation": action_equity_allocation,
                "port_return": portfolio_return}
        state = np.array([self.age, self.wealth], dtype=np.float32)
        return state, reward, done, info

    def render(self, mode='human'):
        # no urgent need for this
        pass

    def reset(self):
        # Resetting age and wealth
        self.wealth = 0
        self.age = self.starting_age
        self.income = self.starting_income
        state = np.array([self.age, self.wealth], dtype=np.float32)
        return state
