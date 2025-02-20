---
layout: default
title: status
---

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
num_mini_batch =  1 \
num_env_steps = 10000000000000 (i am training at 100_000, becasue my computer is very slow)\
ppo_epoch = 15 \
lr = 7e-4 \
critic_lr = 1e-3 \
hidden_size = 512 \
entropy_coef = 0.015 

## Current Progress
Pan:
- **Paper Reproduction**: We have **read and understood the code** from *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, which has provided a solid foundation for our implementation and understanding of the PPO algorithm in cooperative multi-agent settings.
- **Initial Agent Performance**: We have conducted **different experiments** with the PPO agent in various Hanabi setups. Preliminary results show that the agent is able to learn basic strategies, though further tuning and modifications are necessary to enhance its performance in more complex game scenarios.
- **Result Displaying:
   - 2 players
![test1](https://github.com/user-attachments/assets/55fb6aef-32fe-417e-a891-28d64cf6ec76)
   - 3 players
![test2](https://github.com/user-attachments/assets/2d06b883-de92-4c5f-bf27-77fcfe263894)





## Challenges
Pan:
- **Environment Adaptation**: Modifying PPO to fit our custom Hanabi environment while aligning observation and action spaces.  
- **Performance Optimization**: Fine-tuning hyperparameters and model structure to improve learning efficiency.  
- **Team Collaboration**: Ensuring consistency between PPO and A2C while integrating different approaches.  

## Next Steps
Pan:
1. **Integrate PPO** into our Hanabi environment for smooth gameplay.  
2. **Optimize Performance** by refining hyperparameters and model design.  
3. **Compare Algorithms** by benchmarking PPO against A2C and heuristic agents.  
4. **Collaborate & Refine** through team discussions and iterative improvements.  


## References
- [Learning and Reproduction](https://github.com/marlbenchmark/on-policy)
- Use Chatgpt to improve our status report
