from abstract_classes import BaseEnvironment, BaseAgent
import numpy as np

class CliffWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info = {}):
        """
        Initialize the cliffwalking environment
        """
        reward = None
        state = None
        terminal = None
        self.reward_state_terminal = (reward, state, terminal)

        self.grid_h = env_info.get("grid_height", 4)
        self.grid_w = env_info.get("grid_width", 12)

        self.start_loc = (self.grid_h - 1,0)
        self.goal_loc = (self.grid_h-1, self.grid_w-1)
        self.cliff = [(self.grid_h-1, i) for i in range(1, self.grid_w-1)]
        self.state_to_idx_dict = {}
        count = 0
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                self.state_to_idx_dict[(i,j)] = count
                count += 1
    
    def env_start(self):
        """
        Initialize agent location in the environment. Initialize reward = 0, termination = False

        Returns:
            The current state from the environment
        """
        reward = 0
        terminal = False
        self.agent_loc = self.start_loc
        curr_state = self.state(self.agent_loc)
        self.reward_state_terminal = (reward, curr_state, terminal)
        return self.reward_state_terminal[1]

    def env_step(self, action):
        """
        Environment state transition based on action
        
        Args:
        action: The action taken by the agent.

        Returns:
        (state, reward, terminal): A tuple indicating next state, reward based on last action,
        and whether reaches terminal state
        """    
        x,y = self.agent_loc
        # UP
        if action == 0:
            x = x - 1
        # LEFT
        if action == 1:
            y = y -1
        # DOWN
        if action == 2:
            x = x + 1
        # RIGHT
        if action == 3:
            y = y + 1
        # Check if out of boundary
        if not self.isbound(x,y, self.grid_h, self.grid_w):
            x, y = self.agent_loc

        # Default reward per transition is -1
        reward = -1
        terminal = False
        self.agent_loc = (x,y)

        # Check if reaches cliff
        if self.agent_loc in self.cliff:
            reward = -100
            self.agent_loc = self.start_loc
        
        # Check if reaches goal state
        if self.agent_loc == self.goal_loc:
            terminal = True
        
        curr_state = self.state(self.agent_loc)
        self.reward_state_terminal = (reward, curr_state, terminal)
        return self.reward_state_terminal

    def env_cleanup(self):
        """
        Reset the environment once reaches terminal state.
        """
        self.agent_loc = self.start_loc
    
    #### Assisting Function ####
    def state(self,location):
        """
        Map location to state. In this case state is a idx
        |(0, 0) (0, 1)| |0 1|
        |(1, 0) (1, 1)| |2 3|
        """
        return self.state_to_idx_dict[(location)]    
    
    def isbound(self, x,y, height, width):
        """
        Check if the position is in bound

        Returns: Boolean
        """
        if x < 0 or x > (height - 1) or y < 0 or y > (width - 1):
            return False
        else:
            return True

class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        """
        For TD(0) policy evaluation algorithm, the agent takes policy, step size and discount as input
        V(s) should also be initalize
        """
        # Fix random number generator with provided seed
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Note policy is an array of (# States, # Actions)
        self.policy = agent_info.get("policy")
        self.step_size = agent_info.get("step_size")
        self.discount = agent_info.get("discount")
        
        # Initialize V(s)
        self.values = np.zeros((self.policy.shape[0],))

    def agent_start(self, state):
        """
        Pick agent initial action based on the policy and initial state

        Returns:
            The first action agent takes
        """
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.prev_state = state
        return action

    def agent_step(self, reward, state):
        """
        Input is observe reward R and next state S' by taking action A
        Update the value estimation using TD(0) bootstrapping

        Returns:
            The action the action is going to take
        """
        
        td_target = reward + self.discount * self.values[state]
        v_s = self.values[self.prev_state]
        v_s = v_s + self.step_size * (td_target - v_s)
        self.values[self.prev_state] = v_s
        self.prev_state = state

        # Pick next action
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        return action

    def agent_end(self, reward):
        """
        Update the value estimation when reaching terminal state
        """
        td_target = reward
        v_s = self.values[self.prev_state]
        v_s = v_s + self.step_size * (td_target - v_s)
        self.values[self.prev_state] = v_s
        
    def agent_cleanup(self):
        """
        Reset state
        """
        self.prev_state = None

    def agent_message(self, message):
        """
        A function to retreive information from the agent.
        """
        if message == "get_values":
            return self.values
        else:
            raise Exception('Message not understood')

def cliff_walk_policy():
    """
    Define walk along cliff policy, (# of states, # of actions)
    Returns:
        policy grid
    """
    policy_ls = np.ones((env.grid_h * env.grid_w, 4)) * 0.25
    policy_ls[36] = [1,0,0,0]
    for i in range(24, 24 + env.grid_w-1):
        policy_ls[i] = [0,0,0,1]
    policy_ls[35] = [0,0,1,0]
    return policy_ls

env = CliffWalkEnvironment()
agent = TDAgent()


env.env_init({ "grid_height": 4, "grid_width": 12 })
policy_list = cliff_walk_policy()
agent.agent_init({"policy": np.array(policy_list), "discount": 1, "step_size": 0.01})

# Start episode
env.env_start()
reward, state, terminal = env.reward_state_terminal
action = agent.agent_start(state)

value_estimate = agent.agent_message("get_values")
print(value_estimate)
while not terminal:
    reward, state, terminal = env.env_step(action)
    action = agent.agent_step(reward, state)
value_estimate = agent.agent_message("get_values")
print(value_estimate)