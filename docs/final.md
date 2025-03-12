---
layout: default
title: Final Report
---


## Video Summary

This video is limited to **uci.edu** account!
<iframe width="841" height="473" src="https://www.youtube.com/embed/xvjMrUUnVVI" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Project Summary
In this project, we aim to develop AI agents capable of playing **Hanabi** using reinforcement learning techniques. Given the game's **partially observable** and **cooperative** nature, we explore multiple approaches to train agents that can make optimal decisions and achieve high scores. For the environment, we tested both deep-mind learning environment and customized environment. And we have expored different methods to achieve the goal namely, RPPO, Multi-agent PPO, and A2C.


## Approach
Pan: \
After implementing the initial approach, I attempted to further reproduce the methods described in the paper. My goal was to adapt the reproduced code to our custom environment, ensuring that it aligned with our specific settings and constraints. However, the actual results did not meet expectations. Despite careful modifications and integration efforts, the model’s performance remained suboptimal. The agent struggled to effectively coordinate actions and achieve high scores, suggesting that additional adjustments or alternative approaches might be necessary.
![aacd4aafed3263c70837116c1d27f47](https://github.com/user-attachments/assets/f16fbc74-0163-4cf0-9e4e-74a24a5a360c)
After attempting to integrate the reproduced code into our custom environment, I decided to shift to using the original environment from the paper due to compatibility issues, possibly caused by version differences.


To ensure the code could run properly in the original environment, I made modifications to certain parts of the implementation. For example, comparing the two provided code snippets:
<img width="532" alt="image" src="https://github.com/user-attachments/assets/70526fca-2474-4908-bad3-07aaed59bb97" />
<img width="612" alt="image" src="https://github.com/user-attachments/assets/26766006-205c-475d-90b5-2ba843940e3b" />
The second image shows the original code, where self.fc_h and get_clones(self.fc_h, self._layer_N) were used.
However, in the first image, I modified it by replacing get_clones() with nn.ModuleList(), ensuring compatibility with the newer framework while preserving the intended structure.
This adjustment was necessary as the original implementation did not function correctly in the current version of the environment. By making these changes, I was able to execute the code while maintaining its original architecture as closely as possible.

Parameters:\
<img width="600" alt="image" src="https://github.com/user-attachments/assets/ab9b20ab-7e1d-4cc0-9f8d-13ad6d903dd3" />


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
- **With the deadline approaching (two week left), I started training the model. However, upon further examining the GitHub repository, I realized that training a fully optimized model would take an entire month. Given the time constraints, I decided to use the pre-trained model provided in the repository instead of training from scratch.
- This is the part I train but I don't have enough time to train the whole thing.
![result](https://github.com/user-attachments/assets/66df731d-5ca2-4e34-9a36-58ed47363c48)
- This is the final score I used his traind model.
<img width="431" alt="image" src="https://github.com/user-attachments/assets/affdc2c1-17ad-4b50-a255-9b34efa5bdc3" />



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
