---
layout: default
title: status
---


## Video Summary

This video is limited to **uci.edu** account!
<iframe width="841" height="473" src="https://www.youtube.com/embed/xvjMrUUnVVI" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Project Summary
In this project, we aim to develop AI agents capable of playing **Hanabi** using reinforcement learning techniques. Given the game's **partially observable** and **cooperative** nature, we explore multiple approaches to train agents that can make optimal decisions and achieve high scores. For the environment, we tested both deep-mind learning environment and customized environment. And we have expored different methods to achieve the goal namely, RPPO, Multi-agent PPO, and A2C.


## Approach
Pan: 

I aim to train an AI agent using **Proximal Policy Optimization (PPO)** to achieve high scores and effective teamwork in Hanabi’s **partially observable** environment.  
As a cooperative game with limited information, **MAPPO (Multi-Agent PPO)** is a crucial choice, enabling agents to share a **centralized value function** for better coordination. The paper *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* highlights PPO and MAPPO’s success in similar settings, making it an ideal approach. Therefore, I am reproducing its code as the foundation for my project.   

My approach applies **PPO** within a customized **Hanabi** environment, following these steps:  
- **Environment Setup**  
   We use a Hanabi simulator that enforces **partial observability**, allowing agents to see only other players' cards, ensuring adherence to game rules.  

- **PPO Implementation**  
   We adapt **MAPPO** from an existing codebase, aligning it with Hanabi’s cooperative dynamics to enhance coordination.  

- **Training Process**  
   The agent interacts with the environment, receives rewards, and updates its policy using the **PPO objective**, ensuring stable learning. Agents act in turns, communicating indirectly through card-play decisions.  

- **Baseline Comparisons**  
   We evaluate PPO performance across different settings, adjusting **player count, observation constraints, and communication methods** to analyze its adaptability in Hanabi.

Yukai:

My approach is similar as Pan's which is using a multi-agent PPO model described in “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.”, where actor and critic have separated LSTM networks. Following Centralized Training with Decentralized Execution structure, we make the ACTOR network based on each agent’s local observation and the CRITIC network based on both local and global observation. The environment at the beginning is the deep-mind’s Hanabi learning environment, but we also create our own environment for a future customized reward system.  

The sampling is done by every step, we collect the observation, action, reward, predict critic value, and log probability to a buffer. These data will be used to calculate advantage and returns.

My actor and critic is following the equation below: 

<img width="761" alt="image" src="https://github.com/user-attachments/assets/efd4ae72-a94c-4093-a25d-ed90e860c69e" />
<img width="822" alt="image" src="https://github.com/user-attachments/assets/4cf25fc5-f3d0-4b0d-949b-316f8073e339" />

I am taking parameters similar to the research: \
actorLearningRate = 7e-4 \
criticLearningRate = 1e-3 \
num_env_steps = 10000000000000 (i am training at 100_000, becasue my computer is very slow)\
numPlayers = 2 \
clipParam = 0.1 \
ppoEpoch = 15 \
numMiniBatch = 1 \
valueLossCoefficient = 0.5 \
entropyCoefficient = 0.015 \
maximumGradientNorm = 0.5


Tia:

- **Custom Hanabi Environment:** I adapted the Hanabi Learning Environment into a Gym-compatible format, ensuring partial observability and legal move constraints. The RL agent controls seat 0, while other seats are run by a RandomAgent.
- **A2C Implementation:**
   - **Stable Baselines3:** Uses A2C with a vectorized environment (DummyVecEnv) and normalization (VecNormalize).
   - **Network Architecture:** A deeper MLP with layers of size [256, 256, 256, 256] to handle the complexity of Hanabi observations.
   - **Hyperparameters:**
      - Learning Rate: linear schedule from 3×10^(−4) down to 1×10^(−5)
      - n_steps=256
      - γ=0.995
      - Entropy Coefficient α=0.05 for exploration
      - GAE λ=0.95
   - **Training Process:** The agent interacts with the environment, gathers transitions, and updates its policy and value function. Training metrics are tracked in TensorBoard for analysis.

## Evaluation
Pan:
- **Paper Reproduction**: We have **read and understood the code** from *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, which has provided a solid foundation for our implementation and understanding of the PPO algorithm in cooperative multi-agent settings.
- **Initial Agent Performance**: We have conducted **different experiments** with the PPO agent in various Hanabi setups. Preliminary results show that the agent is able to learn basic strategies, though further tuning and modifications are necessary to enhance its performance in more complex game scenarios.
- **Result Displaying:
   - 2 players
![test1](https://github.com/user-attachments/assets/55fb6aef-32fe-417e-a891-28d64cf6ec76)
   - 3 players
![test2](https://github.com/user-attachments/assets/2d06b883-de92-4c5f-bf27-77fcfe263894)


Yukai:

The following are the mean game score gained by every 100 episodes: 

<img width="656" alt="image" src="https://github.com/user-attachments/assets/02e422cf-6de5-4099-bd15-4e69b03014c6" />

The graph shows that it is very unstable. I can predict that in a partially observable game the training will have up and down, but this is too unstable. I do think that there is problems with my hyperparameter values tunning and equation implementation. Although the training steps are very little, I still believe that this MAPPO method could do better than 1.25 game score.


Tia:

- **Initial Agent Performance:**
   - The A2C agent shows signs of learning basic cooperative play, though final scores remain modest.
   - TensorBoard logs reveal policy loss and value loss decreasing over time, indicating learning progress.
   - Entropy remains sufficiently high, suggesting continued exploration but also contributing to variability in results.
- **Result Displaying:**
   - The figure below (from TensorBoard) highlights training metrics such as policy loss, value loss, and explained variance.
     <img width="1021" alt="progress report" src="https://github.com/user-attachments/assets/a7147b68-05ff-408f-bc4d-5b220886e996" />
   - Unstable Fluctuations: As with many partially observable settings, the performance can fluctuate significantly. Further tuning is needed to stabilize learning and improve scores.


## Remaining Goals and Challenges
Pan:
- **Environment Adaptation**: Modifying PPO to fit our custom Hanabi environment while aligning observation and action spaces.  
- **Performance Optimization**: Fine-tuning hyperparameters and model structure to improve learning efficiency.  
- **Team Collaboration**: Ensuring consistency between PPO and A2C while integrating different approaches.

Yukai:
1. The goal for this quarter is to implement a MAPPO model and customized environment that trains agents to play hanabi at perfect score.
2. Also the comparison between different model is also our main goal. We ran baseline models to compare each other's tranining time and the average score they get.
3. The main challenge for me is that the RL is really a new area for me, it is hard for me the start from nowhere and reach a solid goal in RL training.

Tia:
- **Performance Optimization:** Tuning hyperparameters—especially the entropy coefficient, learning rate, and network architecture—will be crucial for improving final scores.
- **Complexity of Hanabi:** The cooperative nature and limited information in Hanabi often require advanced coordination mechanisms, making training more challenging than standard RL tasks.

## Next Steps
Pan:
1. **Integrate PPO** into our Hanabi environment for smooth gameplay.  
2. **Optimize Performance** by refining hyperparameters and model design.  
3. **Compare Algorithms** by benchmarking PPO against A2C and heuristic agents.  
4. **Collaborate & Refine** through team discussions and iterative improvements.

Yukai:
1. My current MAPPO runs with deep-mind hanabi learning env, the next step is to use our own env with customized reward system to better train our agents.
2. Reaching a higher score. Currently mean game score are just 1.25, this is surely not enough.

Tia:
- **Extended Training:** Increase total timesteps and use multiple random seeds to ensure more stable and reproducible learning outcomes.
- **Hyperparameter Tuning:** Systematically adjust parameters (e.g., learning rate, network size, entropy coefficient) to improve stability and convergence.
- **Custom Reward Shaping:** Explore alternative reward structures or additional intermediate signals to help the agent learn more efficiently.

## References
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://github.com/marlbenchmark/on-policy)
- [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/abs/1506.02438)
- Use Chatgpt to proveread status report
- Using LLM to understand the equation from paper, provide implementation ideas, debug code, find related resources, evaluation the performance and suggest a possible why.
