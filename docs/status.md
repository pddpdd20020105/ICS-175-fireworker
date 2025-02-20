---
layout: default
title: status
---

## Project Summary
In this project, we aim to develop AI agents capable of playing **Hanabi** using reinforcement learning techniques. Given the game's **partially observable** and **cooperative** nature, we explore multiple approaches to train agents that can make optimal decisions and achieve high scores.

Pan:  
I aim to train an AI agent using **Proximal Policy Optimization (PPO)** to achieve high scores and effective teamwork in Hanabi’s **partially observable** environment.  
As a cooperative game with limited information, **MAPPO (Multi-Agent PPO)** is a crucial choice, enabling agents to share a **centralized value function** for better coordination. The paper *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* highlights PPO and MAPPO’s success in similar settings, making it an ideal approach. Therefore, I am reproducing its code as the foundation for my project.  



## Approach
Pan: 
My approach applies **PPO** within a customized **Hanabi** environment, following these steps:  
1. **Environment Setup**  
   We use a Hanabi simulator that enforces **partial observability**, allowing agents to see only other players' cards, ensuring adherence to game rules.  

2. **PPO Implementation**  
   We adapt **MAPPO** from an existing codebase, aligning it with Hanabi’s cooperative dynamics to enhance coordination.  

3. **Training Process**  
   The agent interacts with the environment, receives rewards, and updates its policy using the **PPO objective**, ensuring stable learning. Agents act in turns, communicating indirectly through card-play decisions.  

4. **Baseline Comparisons**  
   We evaluate PPO performance across different settings, adjusting **player count, observation constraints, and communication methods** to analyze its adaptability in Hanabi.  



## Current Progress
Pan:
- **Paper Reproduction**: We have **read and understood the code** from *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, which has provided a solid foundation for our implementation and understanding of the PPO algorithm in cooperative multi-agent settings.
- **Initial Agent Performance**: We have conducted **different experiments** with the PPO agent in various Hanabi setups. Preliminary results show that the agent is able to learn basic strategies, though further tuning and modifications are necessary to enhance its performance in more complex game scenarios.
- **Result Displaying**: 
![test1](https://github.com/user-attachments/assets/55fb6aef-32fe-417e-a891-28d64cf6ec76)


## Next Steps

## References
