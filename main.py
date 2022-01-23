from agent.agent import TDAgent, QLearningAgent, ExpectedSarsaAgent
from environment.environment import CliffWalkEnvironment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plot import plot_policy

def optimal_policy(env):
    """
    Define walk along cliff optimal policy, (# of states, # of actions)
    Returns:
        policy grid
    """
    policy_ls = np.ones((env.grid_h * env.grid_w, 4)) * 0.25
    policy_ls[36] = [1,0,0,0]
    for i in range(24, 24 + env.grid_w-1):
        policy_ls[i] = [0,0,0,1]
    policy_ls[35] = [0,0,1,0]
    return policy_ls

def safe_policy(env):
    """
    Define safe policy, (# of states, # of actions)
    Returns:
        policy grid
    """
    policy_ls = np.ones((env.grid_h * env.grid_w, 4)) * 0.25
    # Go up
    for i in range(12,37,12):
        policy_ls[i] = [1, 0, 0, 0]
    # Go right
    for i in range(0, 12):
        policy_ls[i] = [0, 0, 0, 1]
    # Go down
    for i in range(11,36,12):
        policy_ls[i] = [0, 0, 1, 0]
    return policy_ls

def TD_zero(policy, num_episode=5000, plt_step=200):
    """
    Evaluate policy with number of episodes using Tabluar TD(0) algorithm.

    Returns:
        A gif visualization of value estimate.
    """
    env = CliffWalkEnvironment()
    agent = TDAgent()

    env.env_init({ "grid_height": 4, "grid_width": 12 })
    policy_list = policy(env)
    policy_list = policy(env)
    agent.agent_init({"policy": np.array(policy_list), "discount": 1, "step_size": 0.01})

    fig, ax = plt.subplots()
    ims = []

    for i in range(num_episode):
        # Start episode
        state = env.env_start()
        action = agent.agent_start(state)
        reward, state, terminal = env.env_step(action)
        while not terminal:
            value_estimate = agent.agent_message("get_values")
            action = agent.agent_step(reward, state)
            reward, state, terminal = env.env_step(action)
            # For plotting
            if (i+1) % plt_step == 0:
                img = ax.imshow(value_estimate.reshape((env.grid_h, env.grid_w)))
                title = ax.text(0.5,1.05,f"Policy Evaluation on \nSafe Policy Predicted Values, Episode {i+1}", 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
                ims.append([img, title])
        else:
            agent.agent_end(reward)
            env.env_cleanup()
            
    fig.colorbar(img, shrink = 1, aspect = 10, orientation = 'horizontal')
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                    repeat_delay=1000)
    ani.save('./data/readme_gifs/Safe policy.gif', writer='imagemagick', fps=30)

def TD_control(agents_ls, num_episode=5000):
    for i in agents_ls:
        if i == 'QLearning':
            agent = QLearningAgent()
        if i == 'ExpectedSarsa':
            agent = ExpectedSarsaAgent()

        grid_height = 4
        grid_width = 12
        env.env_init({ "grid_height": grid_height, "grid_width": grid_width })
        agent.agent_init({"num_actions": 4, "num_states": grid_height * grid_width, "epsilon": 0.1, "step_size": 0.5, "discount": 1.0})


        for _ in range(num_episode):
            # Start episode
            state = env.env_start()
            action = agent.agent_start(state)
            reward, state, terminal = env.env_step(action)
            while not terminal:
                action = agent.agent_step(reward, state)
                reward, state, terminal = env.env_step(action)
            else:
                agent.agent_end(reward)
                env.env_cleanup()

        optimal_policy_map = np.argmax(agent.q, axis = 1).reshape((env.grid_h, env.grid_w))

        plot_policy(optimal_policy_map, i, env.grid_w, env.grid_h)

if __name__ == "__main__":
    env = CliffWalkEnvironment()
    num_episode = 5000
    
    # TD prediction
    #TD_zero(optimal_policy, num_episode)
    #TD_zero(safe_policy, num_episode)
    
    # TD control
    agents_ls = ['QLearning', 'ExpectedSarsa']
    TD_control(agents_ls, num_episode)