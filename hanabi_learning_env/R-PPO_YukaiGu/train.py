import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mappo_model import createMAPPOModel, MAPPOTrainer, saveModel
from hanabi_learning_environment import rl_env
from hanabi_env import CustomHanabiEnv
from utils import Buffer, convertObservationByPlayer, convertObservation
import argparse

'''
======= Recurrent-MAPPO ======
The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games (Algorithm 1)
==============================
'''
config = {
    "players": 2,
    "random_start_player": True,
    "colors": 5,
    "ranks": 5,
    "hand_size": 5,
    "max_information_tokens": 8,
    "max_life_tokens": 3
}


def trainModel(episodes = 1_000_000, actorLearningRate = 7e-4, numPlayers = 2, clipParam = 0.1,
               ppoEpoch = 5, numMiniBatch = 1, valueLossCoefficient = 0.5,
               entropyCoefficient = 0.015,
               maximumGradientNorm = 0.5, criticLearningRate = 1e-3, gamma = 0.99, lamda = 0.95,
               updateEpisodesNum = 10, useDevice = "cpu", popartBeta = 0.001,
               popartEpsilon = 1e-8, chunkSize = 50, envNum = 1000, maxSteps = 100):
    '''
    Initialize θ, the parameters for policy π
    Initialize φ, the parameters for critic V
    Using Orthogonal initialization (Hu et al., 2020)
    Set learning rate α
    '''
    envs = [rl_env.make("Hanabi-Full", num_players = numPlayers) for _ in range(envNum)]
    # envs = [CustomHanabiEnv(config) for _ in range(envNum)]
    model = createMAPPOModel("Hanabi-Full", numPlayers = numPlayers)
    # Using Orthogonal initialization
    model.orthogonalInitialization()
    model.to(torch.device(useDevice))

    # Initialize θ
    actorParameter = list(model.actorLSTM.parameters()) + list(model.actor.parameters())
    # Initialize φ
    criticParameter = list(model.criticLSTM.parameters()) + list(model.critic.parameters())
    # Set learning rate α
    model.actor_optimizer = optim.Adam(actorParameter, lr = actorLearningRate)
    model.critic_optimizer = optim.Adam(criticParameter, lr = criticLearningRate)

    args = argparse.Namespace(
        clipParam = clipParam,
        ppoEpoch = ppoEpoch,
        numMiniBatch = numMiniBatch,
        valueLossCoefficient = valueLossCoefficient,
        entropyCoefficient = entropyCoefficient,
        maximumGradientNorm = maximumGradientNorm
    )
    trainer = MAPPOTrainer(args, model, device = useDevice)
    buffer = Buffer(numPlayers)

    # Tensorboard
    logDir = f"runs/Hanabi_RMAPPO/Deepmind_End_6"
    writer = SummaryWriter(log_dir = logDir)

    '''
    while step ≤ step_max do
    '''
    for episode in range(episodes):
        '''
        set data buffer D = {}
        for i= 1 to batch_size do
        τ = [] empty list
        '''
        buffer.clear()

        rewardsLog = [0 for _ in range(envNum)]
        stepsLog = [0 for _ in range(envNum)]
        scoresLog = [0 for _ in range(envNum)]

        globalObservations = []
        for i in range(envNum):
            globalObservations.append(envs[i].reset())

        '''
        initialize h(1)_0,π,...h(n)_0,π actor RNN states
        initialize h(1)_0,V,...h(n)_0,V critic RNN states
        '''
        hiddenStates = [[
            ((torch.zeros(1, 1, 512, device = useDevice), torch.zeros(1, 1, 512, device = useDevice)),
            (torch.zeros(1, 1, 512, device = useDevice), torch.zeros(1, 1, 512, device = useDevice))) for _ in range(numPlayers)
            ] for i in range(envNum)]

        dones = [False] * envNum
        stepCount = 0

        '''
        for t = 1 to T do
        '''
        while True:
            stepCount += 1
            allDone = True
            '''
            for all agents a do
            '''
            for i in range(envNum):
                if not dones[i]:
                    allDone = False

                    currentPlayerID = globalObservations[i]['current_player']
                    currentAgentObservation = globalObservations[i]['player_observations'][currentPlayerID]

                    currentAgentObservationVectorized = torch.tensor(convertObservationByPlayer(globalObservations[i], playerID = currentPlayerID),
                                                dtype = torch.float).unsqueeze(0).unsqueeze(0).to(torch.device(useDevice))
                    currentAgentObservationVectorized = currentAgentObservationVectorized.contiguous()
                    globalObservationVectorized = torch.tensor(convertObservation(globalObservations[i]),
                                                dtype = torch.float).unsqueeze(0).unsqueeze(0).to(torch.device(useDevice))
                    globalObservationVectorized = globalObservationVectorized.contiguous()

                    '''
                    p(a)_t ,h(a)_t,π = π(o(a)_t ,h(a)_t−1,π; θ)
                    (Forward actor: use current agent observation, hidden layer, and actor parameters to get action probabilities and new hidden layer)
                    v(a)_t ,h(a)_t,V= V(s(a)_t ,h(a)_t−1,V; φ)
                    (Forward critic: use global observation, hidden layer, and critic parameters to get critic value and new hidden layer)
                    '''
                    actorHiddenLayer, criticHiddenLayer = hiddenStates[i][currentPlayerID]
                    actorHiddenLayer = (actorHiddenLayer[0].detach(), actorHiddenLayer[1].detach())
                    criticHiddenLayer = (criticHiddenLayer[0].detach(), criticHiddenLayer[1].detach())
                    # Forward actor and critic
                    actionProbabilities, newActorHiddenLayer = model.forwardActor(currentAgentObservationVectorized, actorHiddenLayer, device = useDevice)
                    criticValue, newCriticHiddenLayer = model.forwardCritic(globalObservationVectorized, criticHiddenLayer, device = useDevice)
                    # update hidden states
                    hiddenStates[i][currentPlayerID] = (newActorHiddenLayer, newCriticHiddenLayer)

                    '''
                    u(a)_t ∼ p(a)_t
                    (From action probabilities to sample an action)
                    '''
                    legalActions = currentAgentObservation['legal_moves']
                    legalActions = [i.to_dict() if not isinstance(i, dict) else i for i in legalActions]
                    candidateIndex = torch.multinomial(actionProbabilities[0, 0, :], num_samples = 1).item()
                    candidate = envs[i].game.get_move(candidateIndex).to_dict()
                    if candidate in legalActions:
                        action = candidateIndex
                    else:
                        action = next((j for j in range(envs[i].num_moves())
                                       if envs[i].game.get_move(j).to_dict() == legalActions[0]), 0)

                    '''
                    Execute actions u_t, observe r_t, s_t+1, o_t+1
                    '''
                    nextGlobalObservation, reward, done, info = envs[i].step(action)

                    # Get next step info
                    nextPlayerID = nextGlobalObservation['current_player']
                    nextAgentObservation = nextGlobalObservation['player_observations'][nextPlayerID]

                    rewardsLog[i] += reward
                    stepsLog[i] += 1
                    scoresLog[i] = sum(envs[i].state.fireworks())

                    '''
                    τ += [s_t, o_t, h_t,π, h_t,V, u_t, r_t, s_t+1, o_t+1]
                    (Add global observation, current agent observation, Actor Hidden, Critic Hidden,
                    action, reward, next global observation, next agent observation to the buffer)
                    '''
                    buffer.insert(
                        agentID = currentPlayerID,
                        globalObservations = globalObservations[i],
                        currentAgentObservation = currentAgentObservation,
                        actorHidden = newActorHiddenLayer[0].detach().cpu().numpy(),
                        actorCell = newActorHiddenLayer[1].detach().cpu().numpy(),
                        criticHidden = newCriticHiddenLayer[0].detach().cpu().numpy(),
                        criticCell = newCriticHiddenLayer[1].detach().cpu().numpy(),
                        actions = np.array([[action]]),
                        rewards = np.array([[reward]]),
                        nextGlobalObservations = nextGlobalObservation,
                        nextAgentObservations = nextAgentObservation,
                        valuePredict = criticValue.detach().cpu().numpy(),
                        oldPolicyActionLogProbability = torch.log(actionProbabilities[0, 0, action] + 1e-10)
                        .detach().cpu().numpy().reshape(1, 1),
                    )

                    # Update
                    globalObservations[i] = nextGlobalObservation
                    dones[i] = done

            if allDone or stepCount >= maxSteps:
                break

        '''
        Compute advantage estimate A via GAE on τ, using PopArt
        Compute reward-to-go R on τ and normalize with PopArt
        '''
        buffer.computeReturnsAndAdvantages(model = model, gamma = gamma, lamda = lamda,
                                           beta = popartBeta, epsilon = popartEpsilon)

        '''
        TRAIN PART: (The "Split trajectory τ into chunks of length L" is implemented in the Buffer class as sampleMiniBatch())

        for mini-batch k = 1,..., K do
            b ← random mini-batch from D with all agent data
            for each data chunk c in the mini-batch b do
                update RNN hidden states for π and V from first hidden state in data chunk
            end for
        end for
        Adam update θ on L(θ) with data b
        Adam update φ on L(φ) with data b
        '''
        trainer.train(buffer, chunkSize = chunkSize)

        # Output information to tensorboard
        if episode % updateEpisodesNum == 0:
            # avgReward = np.mean(rewardsLog)
            # avgLength = np.mean(stepsLog)
            # avgGameScore = np.mean(scoresLog)
            # writer.add_scalar("Episode Reward", avgReward, episode)
            # writer.add_scalar("Episode Steps", avgLength, episode)
            # writer.add_scalar("Mean Game Score", avgGameScore, episode)
            # print(
            #     f"[Log] Episode {episode}: Steps = {avgLength}, Game Score = {avgGameScore}, Reward = {avgReward}")
            avgScore, avgReward, avgSteps = evaluate(model, config, device = useDevice)
            writer.add_scalar("Evaluation/Average Game Score", avgScore, episode)
            writer.add_scalar("Evaluation/Average Reward", avgReward, episode)
            writer.add_scalar("Evaluation/Average Steps", avgSteps, episode)
            print(
                f"[Evaluation] Episode {episode}: Avg Game Score = {avgScore}, Avg Reward = {avgReward}, Avg Steps = {avgSteps}")

        if episode % 10 == 0:
            saveModel(model, filename = f"RMAPPO_Model.pth")
            print(f"This model is saved at episode {episode}")

    saveModel(model)
    writer.close()


def evaluate(model, config, device, episodes = 10):
    env = CustomHanabiEnv(config)
    scores = []
    rewards = []
    steps = []

    for episode in range(episodes):
        globalObservations = env.reset()
        done = False
        stepCount = 0
        totalReward = 0.0

        numPlayers = config.get("players", 2)
        hiddenStates = []
        for player in range(numPlayers):
            actorHidden = (
            torch.zeros(1, 1, 512, device = device), torch.zeros(1, 1, 512, device = device))
            criticHidden = (
            torch.zeros(1, 1, 512, device = device), torch.zeros(1, 1, 512, device = device))
            hiddenStates.append((actorHidden, criticHidden))

        gameScore = 0
        while not done:
            stepCount += 1
            currentPlayerID = globalObservations['current_player']
            currentObs = globalObservations['player_observations'][currentPlayerID]
            legalMoves = currentObs.get('legal_moves', [])
            legalMoves = [move if isinstance(move, dict) else move.to_dict() for move in legalMoves]

            currentObservation = torch.tensor(
                convertObservationByPlayer(globalObservations, playerID = currentPlayerID),
                dtype = torch.float, device = device).unsqueeze(0).unsqueeze(0)
            currentObservation = currentObservation.contiguous()
            actorHidden, criticHidden = hiddenStates[currentPlayerID]
            actionProb, newActorHidden = model.forwardActor(currentObservation, actorHidden, device)
            candidateIndex = torch.argmax(actionProb[0, 0, :]).item()

            candidate = env.game.get_move(candidateIndex).to_dict()
            if candidate in legalMoves:
                action = candidateIndex
            else:
                legalActionCandidate = legalMoves[0]
                action = next((j for j in range(env.num_moves())
                               if env.game.get_move(j).to_dict() == legalActionCandidate),
                              candidateIndex)

            globalObservations, reward, done, info = env.step(action)
            totalReward += reward
            gameScore = sum(env.state.fireworks())
            hiddenStates[currentPlayerID] = (newActorHidden, criticHidden)

        scores.append(gameScore)
        rewards.append(totalReward)
        steps.append(stepCount)

    averageScore = np.mean(scores)
    averageReward = np.mean(rewards)
    averageSteps = np.mean(steps)

    return averageScore, averageReward, averageSteps


if __name__ == "__main__":
    trainModel()

