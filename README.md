# Reinforcement-Learning-Cliff-Walking

This repo contains python implementation to the cliff walking problem from RL Introduction by Sutton & Barto Example 6.6. <br/>

The purpose is to implement TD(0) policy evaluation and also Q-Learning, Expected Sarsa for policy control. <br/>

## Table of Contents
* [Rules](#rules)
* [RL algorithms](#rl-algorithms)
    * [TD(0)](#td-zero)
    * [Q-Learning](#q-learning)
    * [Expected Sarsa](#expected-sarsa)
* [Setup](#setup)


---
## Rules
<p> <img src="data/readme_pics/rules.JPG"/> </p>
A standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down, <br/>
right, and left. Reward is 1 on all transitions except those into the region marked “The Cli↵.” <br/>
Stepping into this region incurs a reward of 100 and sends the agent instantly back to the start.<br/>

## RL algorithms
### TD Zero
<p> <img src="data/readme_gifs/Optimal policy.gif"/> </p>
<p> <img src="data/readme_gifs/Safe policy.gif"/> </p>

### Q Learning
<p> <img src="data/readme_pics/QLearning_policy_map.jpg"/> </p>

### Expected Sarsa
<p> <img src="data/readme_pics/ExpectedSarsa_policy_map.jpg"/> </p>
