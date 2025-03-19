import numpy as np
import gym
from hanabi_learning_environment import rl_env

class DualAgentHanabiEnv(gym.Env):
    """
    Dual-agent Gym wrapper for Hanabi. Compatible with older Gym API.
    This environment allows training two agents simultaneously:
    - Agent A controls seat 0 in this environment
    - Agent B controls seat 1 in this environment
    """
    def __init__(self, agent_id=0, config=None):
        self.metadata = {'render.modes': ['human']}
        
        if config is None:
            config = {
                "colors": 3,
                "ranks": 3,
                "players": 2,
                "hand_size": 4,
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "observation_type": 1,
            }
        self.config = config
        self.num_players = config["players"]
        self.agent_id = agent_id  # 0 or 1
        
        # Create the underlying Hanabi environment
        self.env = rl_env.HanabiEnv(config=self.config)
        
        # Reset once to determine observation vector size
        obs_dict = self.env.reset()
        example_vec = obs_dict["player_observations"][0]["vectorized"]
        self.obs_dim = len(example_vec)
        
        # Define action and observation spaces - using gym spaces (not gymnasium)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.env.num_moves())
        self.obs_dict = obs_dict
        
        # Store last actions for the other agent (used during evaluation)
        self.other_agent_last_action = None
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        return [seed]
    
    def reset(self):
        """Reset the environment and return the initial observation."""
        self.obs_dict = self.env.reset()
        self.other_agent_last_action = None
        
        # If agent_id is 1 but current player is 0, we need a dummy observation
        if self.agent_id == 1 and self.obs_dict["current_player"] == 0:
            return np.zeros(self.obs_dim, dtype=np.float32)
        else:
            return self._extract_observation()
    
    def step(self, action):
        """Process the action for the current agent."""
        current_player = self.obs_dict["current_player"]
        
        # Default values
        reward = 0
        info = {}
        done = False
        
        # If it's this agent's turn
        if current_player == self.agent_id:
            # Convert action to valid move
            legal_moves = self.obs_dict["player_observations"][current_player]["legal_moves"]
            if len(legal_moves) > 0:
                move_index = action % len(legal_moves)
                action_dict = legal_moves[move_index]
            else:
                action_dict = None
                
            if action_dict is not None:
                # Execute the move
                self.obs_dict, reward, done, info = self.env.step(action_dict)
            
            # Store action for other agent to know what we did
            self.other_agent_last_action = action
        
        # Next observation may be for the other agent, so we may need to return a dummy observation
        if not done and self.obs_dict["current_player"] != self.agent_id:
            next_obs = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            next_obs = self._extract_observation()
        
        # For Hanabi, all agents share the same reward
        return next_obs, float(reward), done, info
    
    def _extract_observation(self):
        """Extract and return the vectorized observation for the current agent."""
        current_player = self.obs_dict["current_player"]
        if current_player == self.agent_id:
            obs_vec = self.obs_dict["player_observations"][current_player]["vectorized"]
            return np.array(obs_vec, dtype=np.float32)
        else:
            # If it's not this agent's turn, return zeros
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    def render(self, mode="human"):
        """Render the environment. Not implemented for Hanabi."""
        pass
    
    def get_current_score(self):
        """Get the current score from fireworks."""
        current_obs = self.obs_dict["player_observations"][0]  # Any player can see the score
        return sum(current_obs["fireworks"].values())
    
    def get_max_possible_score(self):
        """Get the maximum possible score for the current game configuration."""
        return self.config["colors"] * self.config["ranks"]