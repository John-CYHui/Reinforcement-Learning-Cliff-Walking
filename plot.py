from shutil import which
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

def plot_policy(optimal_policy_grid, agent_name, grid_w, grid_h):
    horizontal_min, horizontal_max, horizontal_stepsize = 0, grid_w, 1
    vertical_min, vertical_max, vertical_stepsize = 0, grid_h, -1

    xv, yv = np.meshgrid(np.arange(horizontal_min, horizontal_max, horizontal_stepsize), 
                        np.arange(vertical_max, vertical_min, vertical_stepsize))

    fig, ax = plt.subplots()

    xd, yd = np.gradient(optimal_policy_grid)

    def func_to_vectorize(x, y, optimal_policy_grid):
        # UP
        if optimal_policy_grid == 0:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, 0, 0.35, fc="k", ec="k", head_width=0.1, head_length=0.1)
        # Left
        if optimal_policy_grid == 1:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, -0.35, 0, fc="k", ec="k", head_width=0.1, head_length=0.1)
        # Down
        if optimal_policy_grid == 2:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, 0, -0.35, fc="k", ec="k", head_width=0.1, head_length=0.1)
        # Right
        if optimal_policy_grid == 3:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, 0.35, 0, fc="k", ec="k", head_width=0.1, head_length=0.1)

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)
    vectorized_arrow_drawing(xv, yv, optimal_policy_grid)
    fig.set_size_inches(grid_w, grid_h)
    plt.yticks(np.arange(0,grid_h+0.1,1))
    plt.xticks(np.arange(0,grid_w+0.1,1))
    # Place Start Point
    plt.text(0.4,0.4,'S', fontsize=20, color = 'r')
    # Place Goal Point
    plt.text(grid_w-0.6, 0.4 ,'G', fontsize=20, color = 'g')
    plt.grid(which='major')
    plt.title(f'$\epsilon$-greedy Optimal Policy Learned by {agent_name}')
    #plt.show()
    plt.savefig(f'./data/readme_pics/{agent_name}_policy_map.jpg')