
from rl.rl_environment import RLEnvironment
import random


class RL_AGENT:
    """A small, simple RL agent using an epsilon-greedy policy and a naive Q-like table.

    This is intentionally simple: policy is a mapping from action (or action tuple)
    to reward estimate. Actions may be single ints (when state_size==1) or tuples
    of ints when state_size>1.
    """

    def __init__(self, predict_pipeline, action_space, epsilon=0.9, epsilon_decay=0.99, min_epsilon=0.01, state_size=3):
        self.environment = RLEnvironment(predict_pipeline, action_space, state_size)
        self.action_space = int(self.environment.action_space)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.state_size = int(state_size)
        # policy: key -> estimated value (key is int for single-action or tuple for multi)
        self.policy = {}

    def select_action(self):
        """Select an action using epsilon-greedy.

        Returns either a single int (if state_size==1) or a list of ints of length state_size.
        """
        # Exploration
        if random.random() < self.epsilon:
            # allow repeats; choose action vector
            if self.state_size == 1:
                return random.randrange(self.action_space)
            return [random.randrange(self.action_space) for _ in range(self.state_size)]

        # Exploitation: pick top actions according to policy
        # Build Q-values for individual atomic actions (0..action_space-1)
        q_list = [(a, self.policy.get(a, 0.0)) for a in range(self.action_space)]
        q_list.sort(key=lambda x: x[1], reverse=True)
        if self.state_size == 1:
            return q_list[0][0]

        # For multi-action, take the top `state_size` unique actions
        top_actions = [a for a, _ in q_list[: self.state_size]]
        return top_actions

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_policy(self, action, reward):
        # Normalize key for storage
        key = tuple(action) if isinstance(action, (list, tuple)) else action
        # Simple assignment: store latest observed reward (could be averaged)
        self.policy[key] = float(reward)

    def train(self, episodes=1000):
        for episode in range(int(episodes)):
            state = self.environment.reset()
            done = False
            while not done:
                action = self.select_action()
                next_state, reward, done, info = self.environment.step(action)
                self.update_policy(action, reward)
            self.update_epsilon()
        return self.policy


def get_rl_agent(predict_pipeline, action_space, epsilon=0.9, epsilon_decay=0.99, min_epsilon=0.01, state_size=3, episodes=1000):
    agent = RL_AGENT(predict_pipeline, action_space, epsilon, epsilon_decay, min_epsilon, state_size)
    return agent.train(episodes=episodes)