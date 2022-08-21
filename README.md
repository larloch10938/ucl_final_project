# ucl_final_project

This is the repository used for my final project. My objective is to compare how the traditional solution of the lifecycle program, obtained using dynamic programming, compare to that of a reinforcement learning agent.

In the dynamic_programming folder I recreate and solve the problem using dynamic programming.

In the reinforcement folder I solve the problem training an agent.

In the test folder few test* notebooks are used the explore.

## How to

To run the optimal solution using RL open the ppo_stable notebook in the reinforcement folder. Run it. The notebook will load a few libraries. The main one is lib_for_dqn where the child environment of the gym environment is defined.

### The environment

The environment is defined in the lib_for_dqn library. The environment is a gym environment where the state is the a combination of two continuos states, age and wealth, and the action is a combination of two continuos actions, consumption and investment allocation.

### The agent

The agent is defined in the ppo_stable notebook. The agent is a PPO agent.

## Test folder

The test folder contains some notebooks to explore the environment and the agent. In it I tested different algorithms, for example I coded a DQN version of the agent.