import gym
import numpy as np
from gym import spaces

from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent


class SingleAgentHanabiEnv(gym.Env):
    """
    Single-agent Gym wrapper for multi-player Hanabi:
      - RL agent controls seat 0.
      - Other seats are RandomAgent by default.
      - Uses old Gym API for SB3 compatibility:
         reset() -> obs
         step(action) -> (obs, reward, done, info)
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                "colors": 5,
                "ranks": 5,
                "players": 2,  # 2 total seats
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "observation_type": 1,  # 'card knowledge'
            }
        self.config = config
        self.num_players = config["players"]
        self.rl_player_id = 0

        # Create the underlying Hanabi environment
        self.env = rl_env.HanabiEnv(config=self.config)

        # All other seats: random
        self.other_agents = [
            RandomAgent({"players": self.num_players})
            for _ in range(self.num_players - 1)
        ]

        # Check dimension by resetting once
        obs_dict = self.env.reset()
        example_vec = obs_dict["player_observations"][self.rl_player_id]["vectorized"]
        self.obs_dim = len(example_vec)

        # We define a static action space = self.env.num_moves().
        # The agent can pick any index in [0 .. num_moves()-1].
        # We'll clamp that index to the current legal moves in step().
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.env.num_moves())

        self.obs_dict = obs_dict

    def reset(self):
        """Return only a single observation (old Gym style)."""
        self.obs_dict = self.env.reset()
        obs = self._extract_observation()
        return obs

    def step(self, action):
        """
        1) If it's RL agent's turn, clamp `action` to the number of legal moves, then step.
        2) Let other seats act randomly until seat 0's turn again or game ends.
        Return (obs, reward, done, info).
        """
        current_player = self.obs_dict["current_player"]
        if current_player == self.rl_player_id:
            # Build the *actual* legal moves from the observation
            legal_moves = self.obs_dict["player_observations"][current_player]["legal_moves"]
            # E.g. if legal_moves is length 12, but agent picks 14 => clamp
            if len(legal_moves) > 0:
                move_index = action % len(legal_moves)
                action_dict = legal_moves[move_index]
            else:
                # If somehow no legal moves, just skip
                action_dict = None

            if action_dict:
                obs_dict, reward, done, info = self.env.step(action_dict)
                self.obs_dict = obs_dict
            else:
                # No valid move => no-op
                reward = 0
                done = False
                info = {}
        else:
            # Not RL's turn
            reward = 0
            done = False
            info = {}

        # Let other seats act randomly until it's seat 0's turn again or game ends
        while not done and self.obs_dict["current_player"] != self.rl_player_id:
            cp = self.obs_dict["current_player"]
            # seat 1 => index 0, seat 2 => index 1, ...
            opp_index = cp - 1 if cp > 0 else 0
            opp_obs = self.obs_dict["player_observations"][cp]
            opp_action_dict = self.other_agents[opp_index].act(opp_obs)
            obs_dict, opp_reward, done, opp_info = self.env.step(opp_action_dict)
            self.obs_dict = obs_dict

        obs = self._extract_observation()
        return obs, float(reward), done, info

    def _extract_observation(self):
        """Grab seat 0's vectorized obs as a 1D array."""
        obs_vec = self.obs_dict["player_observations"][self.rl_player_id]["vectorized"]
        obs_arr = np.array(obs_vec, dtype=np.float32)
        return obs_arr

    def render(self, mode="human"):
        pass
