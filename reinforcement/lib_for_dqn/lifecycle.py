import gym
from gym.spaces import Discrete, Dict, Box

class LifecycleEnv(gym.Env):
    def __init__(self):
        # Here we define the choices on consumption and equity allocation
        self.action_space = Dict({"equity_allocation": Discrete(6),
                                  "consumption": Discrete(190)})
        # Set starting variables
        self.wealth = 100
        self.starting_income = 100
        self.income = self.starting_income
        self.starting_age = 19
        self.age = self.starting_age
        self.retirement_age = 65
        self.terminal_age = 115
        self.max_wealth = 10000
        self.wealth_buckets = 500
        # Here we create our observation space
        self.observation_space = Dict({"age": Discrete(self.terminal_age - self.starting_age + 1), "wealth_bucket": Discrete(self.wealth_buckets)})

    def step(self, action):
        reward = 0
        # Income reduces when you retire
        if self.age >= self.retirement_age:
            self.income = 0
        # define market returns
        risk_return = 0.05
        risk_free_return = 0.01
        # transformation
        # print(action)
        action_equity_allocation = action["equity_allocation"] / 5
        action_consumption = action["consumption"] + 10
        # Apply action
        # print(action_equity_allocation)
        portfolio_return = (risk_return * \
            action_equity_allocation) + (risk_free_return * \
            (1 - action_equity_allocation))
        start_wealth = self.wealth
        # Here income is added only after the portfolio return
        self.wealth = (1 + portfolio_return) * (self.wealth -
                                                action_consumption) + self.income
        if self.wealth < 0:
            self.wealth = 0
            reward += -10000
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
        info = {"age": self.age, "start_wealth": start_wealth, "wealth": self.wealth, 
                "consumption": action_consumption, "equity_allocation": action_equity_allocation, "port_return": portfolio_return}
        state = {"age": self.age, "wealth_bucket": min(int(self.wealth/(self.max_wealth + 1) * self.wealth_buckets), self.wealth_buckets - 1)}
        return state, reward, done, info

    def render(self, mode='human'):
        # no urgent need for this
        pass

    def reset(self):
        # Resetting age and wealth
        self.wealth = 100
        self.age = self.starting_age
        self.income = self.starting_income
        state = {"age": self.age, "wealth_bucket": int(self.wealth/self.max_wealth)}
        return state
