from abstract_classes import BaseEnvironment

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
