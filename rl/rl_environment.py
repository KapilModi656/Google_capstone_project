import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from rl.run_rl_with_pipeline import Pipeline
class RLEnvironment:
    def __init__(self,budget):
        self.pipeline = Pipeline()
        self.states = self.pipeline.n_features
        self.feature_names = self.pipeline.feature_names
        self.action_space=budget
    
    def step(self,action):
        if(action.shape[0] != self.states):
            raise ValueError("Action length mismatch with state features")
        observation = np.random.rand(self.states)
        reward = self.pipeline.predict(action)
        done = True
        info = {}
        return observation, reward, done, info
    def reset(self):
        observation = np.random.rand(self.states)
        return observation
