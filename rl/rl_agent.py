import numpy as np
import pandas as pd
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from rl.rl_environment import RLEnvironment
class RLAgent:
    def __init__(self,budget=1000,epsilon=0.99,eps_min=0.01,eps_decay=0.995,gamma=0.95,lr=0.001):
        self.env = RLEnvironment(budget=budget)
        self.feature_names = self.env.feature_names
        self.state_size = self.env.states
        self.action_size = self.env.action_space
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.lr = lr
        self.q={}
    def get_action(self):
        action = np.zeros(self.state_size)
        if np.random.rand() <= self.epsilon:
            action= np.random.randint(0, self.action_size, self.state_size)
            if(action.sum()>self.action_size):
                action = (action / action.sum()) * self.action_size
                action = action.astype(int)
        else:
            if self.q:
                max_action_value = max(self.q.values())
                best_actions = [action for action, value in self.q.items() if value == max_action_value]
                action = best_actions
            else:
                action= np.random.randint(0, self.action_size, self.state_size)
                if(action.sum()>self.action_size):
                    action = (action / action.sum()) * self.action_size
                    action = action.astype(int)
        return action
  
    def learn(self,action,reward):
        action_key = tuple(action)
        old_q_value = self.q.get(action_key, 0.0)
        td_error = reward - old_q_value
        new_q_value = old_q_value + self.lr * td_error
        self.q[action_key] = new_q_value
    def update_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

def main(budget:int=1000,episodes:int=100):
    """
    Main function to run the RL agent for budget allocation.
    Args:
        budget: Total budget to allocate.
        episodes: Number of training episodes.
    Returns:
        A dictionary with allocation results.
    """
    for ep in range(episodes):
        agent = RLAgent(budget=budget)
        state = agent.env.reset()
        done = False
        while not done:
            action = agent.get_action()
            next_state, reward, done, _ = agent.env.step(action)
            agent.learn(action,reward)
            state = next_state
        agent.update_epsilon()
    best_action_key = max(agent.q, key=agent.q.get)
    allocation = dict(zip(agent.feature_names, best_action_key))

    predicted_sales = agent.env.pipeline.predict(best_action_key)
    result = {
        "allocation": allocation,
        "total_budget": budget,
        "predicted_sales": predicted_sales,
    }
    return result