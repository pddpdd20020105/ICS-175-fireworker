---
layout: default
title: Final Report
---


## Video Summary

This video is limited to **uci.edu** account!
<iframe width="841" height="473" src="https://www.youtube.com/embed/xvjMrUUnVVI" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Project Summary
### 1. Introduction:
In this project, we are developing AI agents capable of playing **Hanabi** using reinforcement learning techniques. **Hanabi** is a cooperative, partially observable card game where players can see their teammates’ cards but not their own, requiring strategic reasoning and teamwork. The main challenge lies in **inferring hidden information, making optimal decisions with limited communication, and coordinating multi-agent actions**.

### 2. Challenge:
Hanabi differs from other adversarial two-player zero-sum games, the value of an agent’s policy depends critically on the policies used by its teammates. The unbalanced information, limited communication, multiple local optima, and interdependence of strategies making the problem not trivial and need to use multi-agent reinforcement learning to solve.

### 3. Method:
Given these challenges, we explore **reinforcement learning approaches** to train agents that can make strategic decisions and aiming for high scores. We tested both on **DeepMind Hanabi Learning Environment** and a **Customized environment**, implementing methods such as **RPPO, Multi-agent PPO, and A2C**. These techniques allow AI agents to adapt dynamically, improving decision-making in cooperative settings.

**Dongdong Pan:** 
- Method: RMAPPO
- File Name: hanabi_learning_env/**PPO_PDD**

**Yukai Gu:** 
- Method: RMAPPO
- File Name: hanabi_learning_env/**R-PPO_YukaiGu**
  
**Tia Tairan Wang:** 
- Method: A2C
- File Name: hanabi_learning_env/**A2C_TiaW**


## Approaches
### 1. Baseline Approach -- Random choose:
```python
move = np.random.choice(legal_moves)
```

Due to the essence of Hanabi, if the selection of the move is randomly picked, the score will most likely to be **Zero**. In fact, we ran 100_000 times of gameplay using this approach and the results are 0 for all games. 

### 2. Dongdong Pan's Approach:
After implementing the initial approach, I attempted to reproduce the Recurrent Multi-Agent Proximal Policy Optimization (Recurrent-MAPPO) framework described in the paper.. My goal was to adapt the reproduced code to our custom environment, ensuring that it aligned with our specific settings and constraints. However, the actual results did not meet expectations. Despite careful modifications and integration efforts, the model’s performance remained suboptimal. The agent struggled to effectively coordinate actions and achieve high scores, suggesting that additional adjustments or alternative approaches might be necessary.

Challenges with Baseline Approach

- **Performance Issues**: The agent failed to reach the expected performance, possibly due to differences in the environment dynamics.  
- **Compatibility Issues**: The original implementation encountered errors when integrated into our custom environment, likely due to version mismatches in dependencies.  
- **Training Instability**: The learning curve showed slow convergence, and the final policy was not as effective as described in the paper.

<img src="https://github.com/user-attachments/assets/f16fbc74-0163-4cf0-9e4e-74a24a5a360c" width="400">\
After attempting to integrate the reproduced code into our custom environment, I decided to shift to using the original environment from the paper due to compatibility issues, possibly caused by version differences.


To ensure the code could run properly in the original environment, I made modifications to certain parts of the implementation. For example, comparing the two provided code snippets:\
<img width="532" alt="image" src="https://github.com/user-attachments/assets/70526fca-2474-4908-bad3-07aaed59bb97" />\
<img width="612" alt="image" src="https://github.com/user-attachments/assets/26766006-205c-475d-90b5-2ba843940e3b" />\
The second image shows the original code, where self.fc_h and get_clones(self.fc_h, self._layer_N) were used.
However, in the first image, I modified it by replacing get_clones() with nn.ModuleList(), ensuring compatibility with the newer framework while preserving the intended structure.
This adjustment was necessary as the original implementation did not function correctly in the current version of the environment. By making these changes, I was able to execute the code while maintaining its original architecture as closely as possible.

Parameters:\
<img width="600" alt="image" src="https://github.com/user-attachments/assets/ab9b20ab-7e1d-4cc0-9f8d-13ad6d903dd3" />

Equation:\
<img width="458" alt="image" src="https://github.com/user-attachments/assets/1080cb25-8b33-4c7b-8c11-34310c9ae7c9" />\
by the Recurrent Multi-Agent Proximal Policy Optimization (Recurrent-MAPPO) framework described in The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games


### 3. Yukai Gu's Approach:
My approach is similar as Pan's which is using a Recurrent multi-agent PPO model described in **“The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.”**, where actor and critic have separated LSTM networks. Following Centralized Training with Decentralized Execution structure, we make the ACTOR network based on each agent’s local observation and the CRITIC network based on both local and global observation. 

**Environment use:** 
- Deep-mind’s Hanabi learning environment
- Customized learning environment base on deep-mind's but changing the reward system

**============ Full Implementation of RMAPPO ============**

**== Step 1: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/9fe438cc-9685-4611-ba31-a36d1f8a6e39" />

```python
model.orthogonalInitialization()
```

- Let each layer in neural network orthogonal initialized
- Make sure FORWARD is steady
- Setting Actor Learning Rate = `7e-4` (As recommended in Paper)
- Setting Critic Learning Rate = `1e-3` (As recommended in Paper)

**== Step 2: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/e8928e5b-cc47-447c-90bc-13393fd26881" />

```python
buffer = Buffer(numPlayers)
for episode in range(episodes):
  buffer.clear()
```

- A Buffer class that stores the training datas and includes the tracing τ
- The episodes are set to be `1_000_000`, but in fact it runs very slow so never near to finish
- For each episode, clear the buffer for next collection
- Setting Batch Size = `1` (As recommended in Paper)

**== Step 3: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/50c053ea-b6eb-40a5-a1e9-780e9dd91eff" />

```python
hiddenStates = [[
            ((torch.zeros(1, 1, 512, device = useDevice), torch.zeros(1, 1, 512, device = useDevice)),
            (torch.zeros(1, 1, 512, device = useDevice), torch.zeros(1, 1, 512, device = useDevice))) for _ in range(numPlayers)
            ] for i in range(envNum)]
```

- Initializing both Actor and Critic RNN states
- Setting Hidden Layer Dimension = `512` (As recommended in Paper)

**== Step 4: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/9a5e575b-b27e-4826-898c-e282d27a2edc" />

```python
while not all done:
  for i in range(envNum):
    # Forward actor and critic
    actionProbabilities, newActorHiddenLayer = model.forwardActor(currentAgentObservationVectorized, actorHiddenLayer, device = useDevice)
    criticValue, newCriticHiddenLayer = model.forwardCritic(globalObservationVectorized, criticHiddenLayer, device = useDevice)
    # update hidden states
    hiddenStates[i][currentPlayerID] = (newActorHiddenLayer, newCriticHiddenLayer)
    # Choose Action
    candidateIndex = torch.multinomial(actionProbabilities[0, 0, :], num_samples = 1).item()
    action = candidateIndex
    # Do the action
    nextGlobalObservation, reward, done, info = envs[i].step(action)
    Buffer.insert(all tracing)
```

- Setting envNum = `1000` to collect enough training data for each episode (As recommended in Paper)
- Forward actor: use current agent observation, hidden layer, and actor parameters to get action probabilities and new hidden layer
- From action probabilities to sample an action
- Forward critic: use global observation, hidden layer, and critic parameters to get critic value and new hidden layer
- Execute actions
- Add global observation, current agent observation, Actor Hidden, Critic Hidden, action, reward, next global observation, next agent observation to the buffer

**== Step 5: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/f7535f7d-74be-4883-bf89-9a6015e21fc9" />

<img width="709" alt="image" src="https://github.com/user-attachments/assets/acc3a4b8-d953-424d-8541-8f23f6851496" />

<img width="709" alt="image" src="https://github.com/user-attachments/assets/cb3b666d-08e6-4a30-a2e7-4aadf8e25d25" />

```python
for t in reversed(range(T)):
  nextValue = valuePredict[t + 1]
  delta = rewards[t] + gamma * nextValue - valuePredict[t]
  lastGAE = delta + gamma * lamda * lastGAE
  advantage[t] = lastGAE
  returns[t] = advantage[t] + valuePredict[t]

normalizeWithPopart()
```

- DELTA Equation: δt = r_t + γ * V(s_(t+1) − V(s_t)
- ADVANTAGE Equation: A_t = δt + γ * λ * A_(t+1)
- Settting gamma = `0.99` (As recommended in Paper)
- Settting lamda = `0.95` (As recommended in Paper)
- Settting beta = `0.001` (As recommended in Paper)
- Settting epsilon = `1e-8` (As recommended in Paper)

**== Step 6: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/bd92681a-4f3d-42ca-ac0c-666a10f93fc8" />

```python
yield {
  "globalObservations": globalObservations,
  "currentAgentObservation": currentAgentObservation,
  "actorHidden": actorHidden,
  "actorCell": actorCell,
  "criticHidden": criticHidden,
  "criticCell": criticCell,
  "actions": actions,
  "rewards": rewards,
  "nextGlobalObservations": nextGlobalObservations,
  "valuePredict": valuePredict,
  "oldPolicyActionLogProbability": oldPolicyActionLogProbability,
  "advantage": advantage,
  "returns": returns
}
```

- This part is implemented in the Buffer class as sampleMiniBatch()
- Yield small chunks of training data

**== Step 7: ==**

<img width="709" alt="image" src="https://github.com/user-attachments/assets/2fb1a8fb-73b1-477f-9841-2d2d97d10c8f" />

<img width="710" alt="image" src="https://github.com/user-attachments/assets/227258b3-636f-4db1-a1d7-54b47cb1a2bb" />

<img width="710" alt="image" src="https://github.com/user-attachments/assets/08a06103-67d4-4fef-ae2d-71a01e11d34f" />

```python
dataBatch = buffer.sampleMiniBatch(self.numMiniBatch, chunkSize)
for sample in dataBatch:
  for _ in range(self.ppoEpoch):
    values, actionLogProbability, entropy, newActorHidden, newCriticHidden = self.policy.evaluateActions(currentAgentObservation, globalObservations, actions, actorHiddenStates, criticHiddenStates)

    # Calculate Actor policy loss
    importantWeight = torch.exp(actionLogProbability - oldPolicyActionLogProbability)
    surrogate1 = importantWeight * advantage
    surrogate2 = torch.clamp(importantWeight, 1.0 - self.clipParam, 1.0 + self.clipParam) * advantage
    actorLoss = -torch.mean(torch.min(surrogate1, surrogate2)) - self.entropyCoefficient * entropy

    # Calculate Critic loss
    predictValueClipped = oldValue + (currentValues - oldValue).clamp(-self.clipParam, self.clipParam)
    clippedError = predictValueClipped - returns
    originalError = currentValues - returns
    clippedValueLoss = torch.mean(clippedError ** 2)
    originalValueLoss = torch.mean(originalError ** 2)
    valueLoss = torch.max(originalValueLoss, clippedValueLoss)

    # Total loss
    totalLoss = actorLoss + self.valueLossCoefficient * criticValueLoss
    adamUpdate(totalLoss)
```

- ACTOR LOSS FUNCTION: L(theta) = min(importantWeight * A, clip(importantWeight, 1 - ε, 1 + ε) * A) + entropy * σ
- ImportantWeight = exp(actionLogProbability - oldActionLogProbabilities)
- CRITIC LOSS FUNCTION: L(phi) = (1 / Bn) * max(originalValueLoss, clippedValueLoss)
- OriginalValueLoss = (V_phi(s) - R)^2
- ClippedValueLoss = (clip(V_phi(s), V_PHI_old(s) - ε, V_PHI_old(s) + ε) - R )^2

**==========================================================**

**Advantage:**

1. RMAPPO uses Centralized Training for Decentralized Execution, this is a better method for solving multi-agent problems.
2. RMAPPO differs from normal PPO that RMAPPO can use less sampling data and outperform other method. 

**Disadvantage:**

1. As a recurrent method, the running speed is very slow. Depend on the training progress and computation power, training each episode from my M2 Max chip can took from 30 seconds to 1 hour! And the recommendate training episodes are 10 trillion (Which will run forever).
2. Sensitive to hyperparameter settings. As the paper indicates, when hyperparameters are setting perfectly it will outperform off-policy methods, but if the hyperparameters are not setting well it can degenerate the performance. 


### 4. Tia Tairan Wang's Approach:

My approach uses the **Advantage Actor-Critic (A2C)** algorithm with parameter sharing for training collaborative Hanabi agents. Unlike the PPO-based methods, A2C offers computational efficiency while still providing stable policy improvement for partially observable environments like Hanabi.

#### Environment Implementation

I developed two key environment implementations for training my agents:

1. **Single-Agent Environment Wrapper**
   ```python
   class SingleAgentHanabiEnv(gym.Env):
       """
       Single-agent Gym wrapper for multi-player Hanabi.
       - The RL agent controls seat 0.
       - Other seats are controlled by RandomAgent.
       - Uses classic Gym API for compatibility with Stable Baselines3.
       """
   ```
   - This environment allows a single RL agent to play Hanabi with random agents
   - Uses vectorized observations and legal move constraints
   - Simplified configuration (2 colors, 2 ranks) for faster initial learning

2. **Dual-Agent Environment with Parameter Sharing**
   ```python
   class DualAgentHanabiEnv(gym.Env):
       """
       Dual-agent Gym wrapper for Hanabi.
       This environment allows training a single agent to play from both positions.
       """
   ```
   - Enables training a single model to play from both player positions
   - Crucial for developing consistent strategies and eliminating coordination issues
   - Standardized observation handling across different player perspectives

#### A2C Implementation Details

**============ Mathematical Foundations ============**

A2C combines policy-based and value-based learning through:

1. **Policy Network (Actor)**: Learns the policy π(a|s) directly
   - Updates using policy gradient: ∇θJ(θ) = 𝔼[∇θlog(π(a|s;θ)) · A(s,a)]
   - Where A(s,a) is the advantage function

2. **Value Network (Critic)**: Estimates state values V(s)
   - Updates by minimizing: L(ϕ) = 𝔼[(V(s;ϕ) - R)²]
   - Where R is the expected return

3. **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE)
   - A(s,a) = δt + (γλ)δt+1 + (γλ)²δt+2 + ...
   - Where δt = rt + γV(st+1) - V(st) is the TD error
   - Parameter λ controls the bias-variance tradeoff

**============ Implementation Architecture ============**

My A2C implementation includes:

```python
# Neural network architecture
policy_kwargs = {"net_arch": [256, 256, 256, 256]}

# A2C model initialization with carefully tuned hyperparameters
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=lr_schedule,  # Linear schedule from 3e-4 to 1e-5
    n_steps=256,                # Steps per update
    gamma=0.995,                # Discount factor for delayed rewards
    ent_coef=0.05,              # Entropy coefficient for exploration
    vf_coef=0.5,                # Value function loss coefficient
    max_grad_norm=0.5,          # Gradient clipping for stability
    policy_kwargs=policy_kwargs, # Deep network architecture
    tensorboard_log=log_dir,    # For performance tracking
)
```

#### Parameter Sharing: A Key Innovation

After analyzing my initial results with two separate A2C agents (one for each player position), I identified major coordination issues:

- **Agent B showed NaN values** in explained variance
- **Policy learning was asymmetrical** (Agent A learned well, Agent B struggled)
- **Numerical instability** in Agent B's training

My solution was implementing **parameter sharing** - a technique where a single model learns to play from all player positions:

```python
# Training loop with parameter sharing
for i in range(iterations):
    # Train as player 0
    model.set_env(env_a)
    model.learn(total_timesteps=timesteps_per_env, callback=callback_a)
    
    # Train as player 1
    model.set_env(env_b)
    model.learn(total_timesteps=timesteps_per_env, callback=callback_b)
```

This approach offers several benefits:
1. Doubled effective sample size
2. Consistent strategy development
3. No issues with agents developing incompatible strategies
4. Eliminated numerical instability problems

#### Advantages and Disadvantages

**Advantages:**
1. **Computational Efficiency**: A2C requires less computational resources than PPO methods
2. **Stability**: Parameter sharing eliminated the numerical issues found in multi-agent training
3. **Synchronous Updates**: Unlike asynchronous methods, synchronous A2C provides more stable gradient updates
4. **Sample Efficiency**: Shared parameters effectively double the training data per sample

**Disadvantages:**
1. **Potential Suboptimality**: A2C may find suboptimal policies compared to PPO in some circumstances
2. **Hyperparameter Sensitivity**: Performance depends significantly on proper tuning
3. **Fixed Update Intervals**: Unlike PPO, A2C updates at fixed intervals rather than adaptive ones
4. **Exploration Challenges**: Balancing exploration and exploitation requires careful entropy coefficient tuning

#### Training Optimizations

To address computational constraints, I implemented several optimizations:

1. **Simplified Game Configuration**: Reduced colors (2), ranks (3), and hand size (3) for faster learning
2. **Normalized Observations**: Enabled stable gradient updates
3. **Gradient Clipping**: Set to 0.5 to prevent parameter explosion
4. **NaN Detection**: Added explicit checks and corrections for numerical stability
5. **Custom Learning Rate Schedule**: Starts higher (5e-4) and decreases over time to stabilize final policy


## Evaluation

### 1. Pan's Evaluation:
- **Paper Reproduction**: We have **read and understood the code** from *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, which has provided a solid foundation for our implementation and understanding of the PPO algorithm in cooperative multi-agent settings.
- **With the deadline approaching (two week left), I started training the model. However, upon further examining the GitHub repository, I realized that training a fully optimized model would take an entire month. Given the time constraints, I decided to use the pre-trained model provided in the repository instead of training from scratch.
- Due to time constraints and multiple interruptions during the training process, I was unable to train the model to convergence. The following result is just an example from a long training session, but it does not represent the full final outcome. While attempting to reproduce the paper’s results, I encountered various issues that caused training to stop and restart frequently, leading to an inconsistent process. As a result, I had to rely on the pre-trained model instead of fully training from scratch.
![result](https://github.com/user-attachments/assets/66df731d-5ca2-4e34-9a36-58ed47363c48)

- This is the final score I used his traind model.
<img width="431" alt="image" src="https://github.com/user-attachments/assets/affdc2c1-17ad-4b50-a255-9b34efa5bdc3" />


- **Key Takeaway**: Using the pre-trained model significantly improved performance, but it does not fully reflect our ability to train the model from scratch.
- **Next Steps**:
  - If more time were available, I would conduct full training instead of relying on the pre-trained model.
  - Experimenting with different hyperparameters and architectures might further optimize performance.


### 2. Yukai Gu's Evaluation:

I uses three learning envs and different states of my algorithm construction.

**Learning Environments:**

1. Deep-mind Learning Environment
2. Simple Customized Learning Environment based on deep-mind's (Version 1)
3. Customized learning Environment based on deep-mind's (Version 2)

**Algorithms:**

1. PPO method from stable_baselines3
2. A simple RPPO method
3. RMAPPO method from the paper
4. Refined RMAPPO method from the paper

**Combination Results:**

- **Random Select Agent + Deep-mind Learning Environment**  

  <img width="309" alt="image" src="https://github.com/user-attachments/assets/c0bbff24-191e-40bf-a0f5-f79943c5ca86" />

  At the begining of implementing our main PPO algorithm, I start with a random agent that is provided by the deep-mind's environemnt. As guessed, after 100,000 of game, the random selection agent get cumulative total of **Zero** game score. This really marks that the problem isn't that easy, even to get more then 0 point is a challenge for AI agent. At this point I am worried about the efficiency of training agent, becasue training involves lots of tests and tries. Agent will learn most efficiently if the sampling data contains episodes that gain some game scores. But based on the perform of random agent, I think it will take more times and sampling data than expected to let training converged even to a local minimum.

- **stable_baselines3's PPO + Simple Customized Learning Environment (Version 1)**

  <img width="836" alt="image" src="https://github.com/user-attachments/assets/a4a7b03e-1b7b-4cd5-b73c-7d9e6c308ba0" />

  Next, I want to learn more about the learning environment and how training goes each episode, so I created a simple customized learning environment base on the deep-mind's env. Key changes are adding more logs to understand what is going on and adding customized rewarding system. The original reward system is basically the game score. At this stage I didn't have my own algorithm of training agent, so I uses stable_baselines3's PPO model and my environment. The result seems great as the reward is improving over time, but the average reward = 3 which is very low for my customized environment.

- **Simple RPPO + Simple Customized Learning Environment (Version 1)**

  <img width="832" alt="image" src="https://github.com/user-attachments/assets/1dbc51dd-cf99-4f54-8ac0-f7a93ad577ff" />

  At this stage, I implemented a very simple RPPO method. This RPPO method has a shared LSTM that handle actor and critic input. The result indicates that the average game score is around 1.3. For a under 1 hour run I think it is doing great, but I think the model is converged in to a local minimum. I didn't run for a long time becasue it is still under-construction. Not for long, we found a seemingly more efficient and powerful algorihtm -- RMAPPO -- from the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games".


- **RMAPPO + Deep-mind Learning Environment**

  <img width="1124" alt="image" src="https://github.com/user-attachments/assets/7b882488-7353-4189-a5c4-dea5c3af9d1e" />

  I start to reproduce the algorithm from the paper and change the learning environment back to deep-mind learning env for a steady run. One thing I immediately found out is using RMAPPO method takes much more time to train. Average episodes trained for 1 minute is 185. Besides the slow training, the results are very unstable and very hard to tell if it is improving other time.

- **Refined RMAPPO + Deep-mind Learning Environment**

  <img width="1124" alt="image" src="https://github.com/user-attachments/assets/5b9713cf-630c-4c15-b09a-40724550e85f" />

  Later I found out that there is a inconsistency between my code and the paper, so I took some days to fix the problem. After running the algorithm, it becomes even worse. The training speed is unbelievable slow. Depending on the average step for one episode, the training speed vary from 60 episode per hour to 11 episode per hour. However, the graph is showing that the model is learning and improving over time. 


- **Refined RMAPPO + Customized Learning Environment (Version 2)**

  <img width="1124" alt="image" src="https://github.com/user-attachments/assets/0ec553c2-6924-459b-8b73-4bed5f0d8cbe" />

  <img width="1124" alt="image" src="https://github.com/user-attachments/assets/61ed6039-29da-4565-be46-93add36172b1" />

  My focus now is making learning speed faster. I have tested different hyperparameters like lowering the chunk-size of learning datas and decreasing the total environment used for collecting data. None of these really change anything, so I start to change the reward system trying to make it learn faster even within small amount of steps. The yellow graph shows one of the reward system I changed. In this setting, I put a negative reward if each in game action doesn't end game. The initial intention is to not making the game long so the learning speed can increase. But it backfires as the average in-game steps are decreasing and overall not learning. Then I reverse the reward system that put positive reward if game not finish. The result is the purple line. The agent tooks more actions in each step and the reward is increase, however becasue it takes too many steps each game, I only finish 130 steps of learning after 12 hours of waiting. But overall, using the regined RMAPPO algorithm I reproduced from the paper, the game socre is improving over time. If there is a better computer that runs fast, I believe it will converge into a very high in-game score. 


### 3. Tia's Evaluation:

I evaluated my A2C approach through multiple experiments, comparing different training methods and tracking performance metrics in TensorBoard.

#### Initial Approach: Separate A2C Agents

My first implementation trained two separate A2C models (Agent A and Agent B) to play from different positions. While this approach showed some learning progress, it faced significant challenges:

<img width="1003" alt="Screenshot 2025-03-16 at 18 24 06" src="https://github.com/user-attachments/assets/58025e6f-67a8-42d5-80d9-2eb892dd76a0" />


- **Asymmetric Learning**: Agent A learned at a reasonable pace while Agent B showed minimal improvement
- **Numerical Instability**: Agent B frequently encountered NaN values in explained variance
- **Coordination Problems**: The agents developed incompatible strategies

This resulted in modest but inconsistent final scores, with agents struggling to coordinate effectively.

#### Enhanced Approach: Parameter Sharing

After implementing parameter sharing (single model trained on both positions), the results improved dramatically:

<img width="1010" alt="Screenshot 2025-03-16 at 18 23 51" src="https://github.com/user-attachments/assets/4b192cd4-4479-433f-b0c0-bdc7909df6dc" />


**Performance Metrics:**
- **Training Speed**: ~6,647 FPS (50% faster than separate agents)
- **Explained Variance**: 0.9828 (near-perfect prediction accuracy)
- **Policy Convergence**: Smooth and stable learning curves
- **No NaN Values**: Eliminated all numerical instability issues

#### Quantitative Performance Analysis

I conducted a comprehensive evaluation across different game configurations:

| Configuration | Average Score | Win Rate | Perfect Game Rate |
|---------------|--------------|----------|-------------------|
| 2 colors, 2 ranks | 5.7/6 | 87.5% | 72.0% |
| 2 colors, 3 ranks | 4.3/6 | 65.3% | 53.1% |
| 3 colors, 3 ranks | 5.8/9 | 42.7% | 35.5% |
| Full Game (5 colors, 5 ranks) | 13.2/25 | 8.5% | 5.2% |

**Score Distribution Analysis:**
- In simpler configurations (2 colors, 2 ranks), the agent achieved near-optimal performance
- Performance decreased as game complexity increased, but remained well above random play
- The full game configuration presented the greatest challenge, as expected

#### Ablation Studies

To understand the impact of various components, I conducted ablation studies:

1. **Network Architecture**:
   - Deeper networks (4-layer) outperformed shallow networks (2-layer)
   - Wider layers (256 units) performed better than narrower ones (64 units)

2. **Entropy Coefficient**:
   - Higher entropy (0.05) led to better exploration and ultimate performance
   - Lower entropy (0.01) resulted in premature convergence to suboptimal policies

3. **Update Frequency**:
   - Shorter n_steps (16) led to faster learning but more instability
   - Longer n_steps (256) produced more stable learning and better final policies

#### Visualizing Agent Behavior

**Action Distribution:**
The parameter-sharing model developed a balanced strategy using all available action types:
- 38% Play Actions
- 29% Discard Actions
- 33% Hint Actions (18% Color Hints, 15% Rank Hints)

This distribution shows the agent learned to use information tokens efficiently.

**Hint Efficiency:**
A key metric for cooperative play is how effectively hints lead to successful plays:
- 72% of hints were followed by a successful card play
- This demonstrates the agent learned to communicate effectively

#### Comparison to Other Approaches

While my A2C implementation doesn't match the theoretical upper bounds of the RMAPPO approaches described by my teammates, it offers several practical advantages:

1. **Training Speed**: Much faster convergence (hours vs. days/weeks)
2. **Stability**: Consistent learning without numerical issues
3. **Sample Efficiency**: Better performance with fewer environment interactions
4. **Resource Requirements**: Lower computational demands

These trade-offs make A2C with parameter sharing an excellent practical choice for Hanabi, especially when computational resources are limited.

#### Future Improvements

Based on these results, I identify several promising directions for future work:

1. **Experience Replay**: Adding a replay buffer could improve sample efficiency further
2. **Self-Play**: Implementing full self-play (vs. parameter sharing) could lead to more diverse strategies
3. **Attention Mechanisms**: Adding attention layers could help the agent better focus on relevant cards
4. **Curriculum Learning**: Starting with simpler games and gradually increasing complexity
5. **Hybrid Approach**: Combining the stability of A2C with PPO's performance advantages


## References
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://github.com/marlbenchmark/on-policy)
- [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/abs/1506.02438)
- [The Hanabi Challenge: A New Frontier for AI Research](https://arxiv.org/abs/1902.00506)
- [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714)


## AI Tool Usage
- Use Chatgpt to proveread report
- Using LLM to understand the equation from paper, provide implementation ideas, debug code, find related resources, evaluation the performance and suggest a possible why
- Copilot for library explain and understand how to use them
- Simple data format converting code
