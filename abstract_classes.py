from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    @abstractmethod
    def env_init(self, env_info = {}):
        pass
    @abstractmethod
    def env_start(self):
        pass
    @abstractmethod
    def env_step(self, action):
        pass
    @abstractmethod
    def env_cleanup(self):
        pass

class BaseAgent(ABC):
    @abstractmethod
    def agent_init(self, agent_info={}):
        pass
    @abstractmethod
    def agent_start(self, state):
        pass
    @abstractmethod
    def agent_step(self, reward, state):
        pass
    @abstractmethod
    def agent_end(self, reward):
        pass
    @abstractmethod
    def agent_cleanup(self):
        pass
    @abstractmethod
    def agent_message(self):
        pass