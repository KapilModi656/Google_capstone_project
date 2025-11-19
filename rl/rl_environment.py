class RLEnvironment:
    """Simple RL environment wrapper around a prediction pipeline.

    predict_pipeline: callable that accepts an observation (list/2D-list)
                      and returns a list-like prediction (e.g. [value]).
    action_space: int, number of discrete actions (0..action_space-1)
    state_size: int, number of values the environment/action expects
    """

    def __init__(self, predict_pipeline, action_space, state_size):
        self.predict_pipeline = predict_pipeline
        self.action_space = int(action_space)
        self.state_size = int(state_size)

    def step(self, action):
        """Take an action and return (observation, reward, done, info).

        Accepts either a scalar (int) when state_size==1, or an iterable
        (list/tuple/ndarray) with length == state_size.
        """
        # Normalize action into a flat list of length state_size
        if self.state_size == 1 and isinstance(action, (int, float)):
            obs = [action]
        else:
            try:
                if len(action) != self.state_size:
                    raise ValueError("Action size does not match state size")
                obs = list(action)
            except TypeError:
                raise ValueError("Action must be an int (for state_size==1) or an iterable of length state_size")

        # The predict_pipeline may expect a 2D shape: [obs]
        try:
            prediction = self.predict_pipeline(obs if hasattr(obs[0], '__len__') else [obs])
        except Exception:
            # try passing as a single observation
            prediction = self.predict_pipeline([obs])

        # Ensure prediction is a list-like and extract scalar reward
        try:
            reward = float(prediction[0])
        except Exception:
            # If not convertible, return 0.0 as fallback
            reward = 0.0

        done = True
        info = {}
        # For simplicity, return the raw prediction as observation
        return prediction, reward, done, info

    def reset(self):
        """Return an initial (zero) state observation."""
        return [0.0] * self.state_size