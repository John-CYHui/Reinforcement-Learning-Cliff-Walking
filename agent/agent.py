
from cmath import exp
from abstract_classes import BaseAgent
import numpy as np
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

class QLearningAgent(BaseAgent):
    def agent_init(self, agent_info={}):
            """
            For Q-learning off-policy TD control algorithm, the agent takes epsilon, step size and discount as input
            Q(s, a) should also be initalize
            """
            # Fix random number generator with provided seed
            self.rand_generator = np.random.RandomState(agent_info.get("seed"))
            
            self.epsilon = agent_info.get("epsilon")
            self.step_size = agent_info.get("step_size")
            self.discount = agent_info.get("discount")
            self.num_states = agent_info.get("num_states")
            self.num_actions = agent_info.get("num_actions")

            # Initialize Q(s,a) (# states, # actions)
            self.q = np.zeros((self.num_states,self.num_actions))

    def agent_start(self, state):
        """
        Pick agent initial action based epsilon greedy policy and initial state

        Returns:
            The first action agent takes
        """
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state,:])

        self.prev_state = state
        self.prev_action = action
        return self.prev_action

    def agent_step(self, reward, state):
        """
        Input is observe reward R and next state S' by taking action A
        Update the value estimation using Q-learning

        Returns:
            The action the action is going to take
        """
        # Epsilon greedy policy
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state,:])
        
        td_target = reward + self.discount * np.max(self.q[state,:])
        q_s_a = self.q[self.prev_state, self.prev_action]
        q_s_a = q_s_a + self.step_size * (td_target - q_s_a)
        self.q[self.prev_state, self.prev_action] = q_s_a

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        """
        Update the value estimation when reaching terminal state
        """
        td_target = reward
        q_s_a = self.q[self.prev_state, self.prev_action]
        q_s_a = q_s_a + self.step_size * (td_target - q_s_a)
        self.q[self.prev_state, self.prev_action] = q_s_a
        
    def agent_cleanup(self):
        """
        Reset state
        """
        self.prev_state = None

    def agent_message(self):
        pass

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)


class ExpectedSarsaAgent(BaseAgent):
    def agent_init(self, agent_info={}):
            """
            For Q-learning off-policy TD control algorithm, the agent takes epsilon, step size and discount as input
            Q(s, a) should also be initalize
            """
            # Fix random number generator with provided seed
            self.rand_generator = np.random.RandomState(agent_info.get("seed"))
            
            self.epsilon = agent_info.get("epsilon")
            self.step_size = agent_info.get("step_size")
            self.discount = agent_info.get("discount")
            self.num_states = agent_info.get("num_states")
            self.num_actions = agent_info.get("num_actions")

            # Initialize Q(s,a) (# states, # actions)
            self.q = np.zeros((self.num_states,self.num_actions))

    def agent_start(self, state):
        """
        Pick agent initial action based epsilon greedy policy and initial state

        Returns:
            The first action agent takes
        """
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state,:])

        self.prev_state = state
        self.prev_action = action
        return self.prev_action

    def agent_step(self, reward, state):
        """
        Input is observe reward R and next state S' by taking action A
        Update the value estimation using Expected Sarsa

        Returns:
            The action the action is going to take
        """
        # Epsilon greedy policy
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state,:])
        
        greedy_action = self.argmax(self.q[state,:])
        # Compute expected q using epsilon greedy
        expected_q = 0.0
        for a in range(self.num_actions):
            if a == greedy_action:
                expected_q = expected_q + (1 - self.epsilon + self.epsilon / self.num_actions) * self.q[state, a]
            else:
                expected_q = expected_q + self.epsilon / self.num_actions * self.q[state, a]

        td_target = reward + self.discount * expected_q
        q_s_a = self.q[self.prev_state, self.prev_action]
        q_s_a = q_s_a + self.step_size * (td_target - q_s_a)
        self.q[self.prev_state, self.prev_action] = q_s_a

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        """
        Update the value estimation when reaching terminal state
        """
        td_target = reward
        q_s_a = self.q[self.prev_state, self.prev_action]
        q_s_a = q_s_a + self.step_size * (td_target - q_s_a)
        self.q[self.prev_state, self.prev_action] = q_s_a
        
    def agent_cleanup(self):
        """
        Reset state
        """
        self.prev_state = None

    def agent_message(self):
        pass
    
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)