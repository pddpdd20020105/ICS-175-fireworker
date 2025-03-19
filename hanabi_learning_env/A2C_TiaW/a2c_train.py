#!/usr/bin/env python3
"""
Fixed Hanabi training script with parameter sharing and stability improvements.
Compatible with older PyTorch versions.
"""
import os
import numpy as np
import torch
import datetime
import time
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment class (adjust if needed)
from dual_agent_hanabi_env import DualAgentHanabiEnv

# Custom callback for monitoring both agents
class MonitorCallback(BaseCallback):
    def __init__(self, eval_env, agent_model, eval_freq=50000, n_eval_episodes=20, verbose=1):
        super(MonitorCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.agent_model = agent_model
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_score = 0
        self.agent_id = eval_env.agent_id
        
    def _on_step(self):
        # Check for NaN values in model parameters
        for param in self.model.policy.parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN detected in model parameters!")
                # Replace NaN with zeros
                param.data[torch.isnan(param)] = 0.0
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            # Evaluate performance
            scores = self.evaluate_agents()
            mean_score = np.mean(scores)
            max_possible = self.eval_env.get_max_possible_score()
            
            # Log metrics
            self.logger.record(f"eval/mean_score_agent{self.agent_id}", mean_score)
            self.logger.record(f"eval/normalized_score_agent{self.agent_id}", mean_score / max_possible)
            self.logger.record(f"eval/win_rate_agent{self.agent_id}", np.mean([s == max_possible for s in scores]))
            
            # Save best model
            if mean_score > self.best_mean_score:
                self.best_mean_score = mean_score
                self.agent_model.save(f"best_agent_{self.agent_id}_{mean_score:.1f}")
                
            print(f"\nAgentID {self.agent_id} Eval ({self.n_calls} steps): Score={mean_score:.1f}/{max_possible}")
                
        return True
    
    def evaluate_agents(self):
        """Evaluate agent performance"""
        scores = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                # Get current player
                current_player = self.eval_env.obs_dict["current_player"]
                # Use same model for both positions (parameter sharing)
                action, _ = self.agent_model.predict(obs, deterministic=True)
                obs, _, done, _ = self.eval_env.step(action)
            
            # Record final score
            scores.append(self.eval_env.get_current_score())
            
        return scores

# Create simplified Hanabi environment
def create_env(agent_id=0):
    config = {
        "colors": 2,
        "ranks": 3,
        "players": 2,
        "hand_size": 3,
        "max_information_tokens": 3,
        "max_life_tokens": 2,
        "observation_type": 1,
    }
    env = DualAgentHanabiEnv(agent_id=agent_id, config=config)
    # Set seed for reproducibility
    env.seed(42 + agent_id)
    return env

def make_env(agent_id):
    def _init():
        return create_env(agent_id)
    return _init

def main():
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Set PyTorch threads
    torch.set_num_threads(1)  # Single-threaded for deterministic behavior
    
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"fixed_training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environments (one for each agent role)
    n_envs = 2  # Keep number small for stability
    
    print("Creating environments...")
    env_a = DummyVecEnv([make_env(agent_id=0) for _ in range(n_envs)])
    env_b = DummyVecEnv([make_env(agent_id=1) for _ in range(n_envs)])
    
    # Normalize observations but not rewards (prevent normalization issues)
    env_a = VecNormalize(env_a, norm_obs=True, norm_reward=False, clip_obs=5.0)
    env_b = VecNormalize(env_b, norm_obs=True, norm_reward=False, clip_obs=5.0)
    
    # Evaluation environments
    eval_env_a = create_env(agent_id=0)
    eval_env_b = create_env(agent_id=1)
    
    # Define network architecture
    policy_kwargs = {
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    }
    
    # Create A2C model with careful initialization
    print("Creating A2C model...")
    model = A2C(
        policy="MlpPolicy",
        env=env_a,  # Start with environment A
        learning_rate=5e-4,  # Slightly lower learning rate for stability
        n_steps=16,  # Smaller batch size for more frequent updates
        gamma=0.99,  # Discount factor
        ent_coef=0.02,  # Slightly higher entropy for better exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping to prevent explosion
        use_rms_prop=False,  # Use Adam optimizer for better stability
        tensorboard_log=log_dir,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    
    # Parameter sharing: We'll use the same model for both agents
    # Create evaluation callbacks
    callback_a = MonitorCallback(
        eval_env=eval_env_a,
        agent_model=model,
        eval_freq=50000
    )
    
    callback_b = MonitorCallback(
        eval_env=eval_env_b,
        agent_model=model,
        eval_freq=50000
    )
    
    # Training settings
    timesteps_per_env = 100000
    iterations = 5
    
    print(f"Starting training with parameter sharing - {iterations} iterations of {timesteps_per_env} steps each")
    
    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        # Train in environment A (agent_id=0)
        print(f"Training as agent 0 for {timesteps_per_env} steps...")
        model.set_env(env_a)
        model.learn(
            total_timesteps=timesteps_per_env,
            callback=callback_a,
            tb_log_name="A2C_shared",
            reset_num_timesteps=False if i > 0 else True
        )
        
        # Train in environment B (agent_id=1)
        print(f"Training as agent 1 for {timesteps_per_env} steps...")
        model.set_env(env_b)
        model.learn(
            total_timesteps=timesteps_per_env,
            callback=callback_b,
            tb_log_name="A2C_shared",
            reset_num_timesteps=False
        )
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_games = 100
    scores_a = []
    scores_b = []
    
    # Evaluate as agent 0
    for _ in range(eval_games):
        obs = eval_env_a.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env_a.step(action)
        scores_a.append(eval_env_a.get_current_score())
    
    # Evaluate as agent 1
    for _ in range(eval_games):
        obs = eval_env_b.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env_b.step(action)
        scores_b.append(eval_env_b.get_current_score())
    
    # Calculate statistics
    mean_score_a = np.mean(scores_a)
    mean_score_b = np.mean(scores_b)
    max_possible = eval_env_a.get_max_possible_score()
    win_rate_a = np.mean([s == max_possible for s in scores_a])
    win_rate_b = np.mean([s == max_possible for s in scores_b])
    
    print("\n===== Final Results =====")
    print(f"Agent 0 - Average Score: {mean_score_a:.2f}/{max_possible} ({mean_score_a/max_possible:.2%})")
    print(f"Agent 0 - Win Rate: {win_rate_a:.2%}")
    print(f"Agent 1 - Average Score: {mean_score_b:.2f}/{max_possible} ({mean_score_b/max_possible:.2%})")
    print(f"Agent 1 - Win Rate: {win_rate_b:.2%}")
    print("==========================")
    
    # Save final model
    model.save("final_shared_agent")
    env_a.save("final_norm_stats_a.pkl")
    env_b.save("final_norm_stats_b.pkl")
    
    print(f"Training complete! Model saved as 'final_shared_agent'")
    print(f"TensorBoard logs saved to: {log_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        # Print full traceback
        import traceback
        traceback.print_exc()