---
layout: default
title: Proposal
---

# {{ page.title }}: Hanabi AI


## Summary of the Project
This project aims to develop an intelligent agent using **reinforcement learning** to achieve a high or perfect score in the game of **HANABI**. In this game, players cannot see their own cards and must make decisions based on clues provided by their teammates, making it a **Partially observable Markov decision process (POMDP)**. The main challenge and objective of this project are to train agents to collaborate efficiently and make optimal decisions under limited information.

The **input** of this project consists of the game state, which includes clues from other players, teammates’ cards, previous actions, the number of remaining clues, the discard pile, and other relevant information of the game. The **output** will be the action done by the agent on its turn which includes discarding a card, giving a clue to one teammate, or playing a card.

This project focuses on **Multi-agent reinforcement learning (MARL)** and the cooperation between agents. The approach can be applied to other POMDP-based games, such as Werewolf (Mafia) or Avalon. Also this project can help further research in cooperative AI systems.
<br>
<br>

## AI/ML Algorithms
We plan to use **reinforcement learning (RL)** in this project. Specifically, we will explore two different approaches: **Proximal Policy Optimization (PPO)** and **Advantage Actor-Critic (A2C)** to train our agents. Our approach is primarily **model-free**, and we will experiment with **on-policy methods (PPO, A2C)** to evaluate their performance.
<br>
<br>

## Evaluation Plan
**Quantitative Evaluation:**  
We will use the **average game score** as the main metric to evaluate our project. The learning environment includes a simple episode runner using the RL environment, where the average score is near 0, indicating that without collaboration, even achieving 1 point is highly challenging. Therefore, we will use this as our baseline. Our goal is for our RL agents to achieve an average score of at least 15 over 25. Additionally, we will track the number of meaningful clues provided by the agent to assess its ability to cooperate effectively.

**Qualitative Evaluation:**  
For qualitative evaluation, we aim to analyze how the AI makes decisions. We will design toy examples with predefined optimal actions and test whether the AI selects the best possible action. The key objective is to determine whether the agents are truly learning to collaborate.

**Moonshot Case:**  
Our ultimate goal is to develop AI agents that can achieve human-level collaboration in Hanabi. Ideally, perform the same or even better to human players.
<br>
<br>

## Meet the Instructor
Group meeting at discord. [01/23/2025] \
Meet with the instructor. [01/28/2025 - 12:10pm] \
Meet with the instructor. [02/03/2025 - 13:10pm] \
<br>
<br>

## AI Tool Usage
Using Chatgpt for brainstorming the ideas. \
Proof read on the proposal. \
Debugging errors.
<br>
<br>
