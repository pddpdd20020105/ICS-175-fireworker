import torch
import torch.nn as nn
import torch.nn.functional as F
from hanabi_env import CustomHanabiEnv as HanabiEnv
from hanabi_learning_environment import rl_env
from utils import convertObservationByPlayer
import numpy as np

"""
ACTOR and CRITIC have their own LSTM
ACTOR outputs action probabilities
CRITIC outputs state value
"""

class MAPPOModel(nn.Module):
    def __init__(self, inputDimension, hiddenDimension, actionDimension):
        super(MAPPOModel, self).__init__()
        # Centralized Training with Decentralized Execution (CTDE)
        self.actorLSTM = nn.LSTM(inputDimension, hiddenDimension, batch_first=True)
        self.criticLSTM = nn.LSTM(inputDimension, hiddenDimension, batch_first=True)

        # ACTOR output
        self.actor = nn.Linear(hiddenDimension, actionDimension)
        # CRITIC output
        self.critic = nn.Linear(hiddenDimension, 1)

    def orthogonalInitialization(self):
        for module in self.modules():
            # If is fully connected layers
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                # Clean bias
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # If is LSTM
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name or "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forwardActor(self, currentAgentObservation, actorHiddenLayer, device):
        # forward ACTOR
        currentAgentObservation = currentAgentObservation.to(device)
        actorOutput, newActorHiddenLayer = self.actorLSTM(currentAgentObservation, actorHiddenLayer)
        actionProbabilities = F.softmax(self.actor(actorOutput), dim=-1)

        return actionProbabilities, newActorHiddenLayer

    def forwardCritic(self, globalObservation, criticHiddenLayer, device):
        # forward CRITIC
        globalObservation = globalObservation.to(device)
        criticOutput, newCriticHiddenLayer = self.criticLSTM(globalObservation, criticHiddenLayer)
        criticValue = self.critic(criticOutput)

        return criticValue, newCriticHiddenLayer

    def popartUpdateCritic(self, oldMean, oldVar, newMean, newVar):
        self.critic.weight.data.mul_(np.sqrt(oldVar) / np.sqrt(newVar))
        self.critic.bias.data.mul_(np.sqrt(oldVar)).add_(oldMean - newMean).div_(np.sqrt(newVar))

    def evaluateActions(self, currentAgentObservation, globalObservations, actions, actorHiddenStates, criticHiddenStates):
        # ======= ACTOR =======
        currentAgentObservation = currentAgentObservation.unsqueeze(0)
        actorOutput, newActorHidden = self.actorLSTM(currentAgentObservation, actorHiddenStates)
        logits = self.actor(actorOutput)
        logProbability = F.log_softmax(logits, dim = -1)

        # Current Action Log Probability
        actions = actions.unsqueeze(0)
        actions = actions.squeeze(2)
        actions = actions.long()

        actionLogProbabilityAll = logProbability.gather(-1, actions).squeeze(-1)
        actionLogProbability = actionLogProbabilityAll.mean()

        probability = logProbability.exp()
        entropyAll = -(probability * logProbability).sum(dim = -1)
        entropy = entropyAll.mean()

        # ======= CRITIC =======
        globalObservations = globalObservations.unsqueeze(0)
        criticOutput, newCriticHidden = self.criticLSTM(globalObservations, criticHiddenStates)
        valueSequence = self.critic(criticOutput)
        values = valueSequence.mean(dim = 1).squeeze(-1)

        return values, actionLogProbability, entropy, newActorHidden, newCriticHidden

def createMAPPOModel(envChoose = "Hanabi-Full", numPlayers = 2):
    # Deep-mind env
    if envChoose == "Hanabi-Full":
        env = rl_env.make("Hanabi-Full", num_players = numPlayers)
        observation = env.reset()
        inputDimension = len(convertObservationByPlayer(observation, playerID = 0))
        hiddenDimension = 512
        actionDimension = env.num_moves()
        model = MAPPOModel(inputDimension, hiddenDimension, actionDimension)
        return model
    # Custom env
    else:
        env = HanabiEnv()
        observation, globalObservation = env.reset()

        inputDimension = len(observation)
        hiddenDimension = 512
        actionDimension = env.actionSpace.n
        model = MAPPOModel(inputDimension, hiddenDimension, actionDimension)
        return model

def saveModel(model, filename="mappo_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"RMAPPO Model saved to {filename}")

def loadModel(filename="mappo_model.pth"):
    model = createMAPPOModel()
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"MAPPO Model loaded from {filename}")
    return model


class MAPPOTrainer:
    # Run on cpu or mps， use mps is little-bit faster
    # MPS is for m-series chip on Macbook
    def __init__(self, args, policy, device = torch.device("mps")):
        self.device = device
        self.tpdv = dict(dtype = torch.float32, device = device)
        self.policy = policy

        '''
        ===== PPO Parameter =====
        clipParam: ε, epsilon, prevent updating too much
        ppoEpoch: repeat number of training on the same data
        numMiniBatch: number of mini-batch
        '''
        self.clipParam = args.clipParam
        self.ppoEpoch = args.ppoEpoch
        self.numMiniBatch = args.numMiniBatch
        self.valueLossCoefficient = args.valueLossCoefficient
        self.entropyCoefficient = args.entropyCoefficient
        self.maximumGradientNorm = args.maximumGradientNorm

    '''
    ==== CRITIC LOSS FUNCTION ====
    L(phi) = (1 / Bn) * max(originalValueLoss, clippedValueLoss)
    originalValueLoss = (V_phi(s) - R)^2
    clippedValueLoss = (clip(V_phi(s), V_PHI_old(s) - ε, V_PHI_old(s) + ε) - R )^2

    The Surprising Effectiveness of PPO in Cooperative
    Multi-Agent Games (Page. 12)
    ==============================
    currentValues: V_Phi(s)
    oldValue: V_PHI_old(s)
    returns: R
    self.clipParam: ε
    '''
    def calculateCriticValueLoss(self, currentValues, oldValue, returns):
        # Clip
        predictValueClipped = oldValue + (currentValues - oldValue).clamp(-self.clipParam, self.clipParam)
        clippedError = predictValueClipped - returns
        originalError = currentValues - returns
        # Loss MSE
        clippedValueLoss = torch.mean(clippedError ** 2)
        originalValueLoss = torch.mean(originalError ** 2)

        # use larger loss
        valueLoss = torch.max(originalValueLoss, clippedValueLoss)
        return valueLoss


    def policyUpdate(self, sample):
        # Unzip mini-batch data and transfer
        globalObservations = torch.from_numpy(np.array(sample["globalObservations"])).to(torch.device(self.device))
        currentAgentObservation = torch.from_numpy(np.array(sample["currentAgentObservation"])).to(
            torch.device(self.device))
        actorHidden = torch.from_numpy(np.array(sample["actorHidden"])).to(
            torch.device(self.device))
        actorCell = torch.from_numpy(np.array(sample["actorCell"])).to(
            torch.device(self.device))
        criticHidden = torch.from_numpy(np.array(sample["criticHidden"])).to(
            torch.device(self.device))
        criticCell = torch.from_numpy(np.array(sample["criticCell"])).to(
            torch.device(self.device))
        actions = torch.from_numpy(np.array(sample["actions"])).to(
            torch.device(self.device))
        valuePredict = torch.from_numpy(np.array(sample["valuePredict"])).to(
            torch.device(self.device))
        oldPolicyActionLogProbability = torch.from_numpy(np.array(sample["oldPolicyActionLogProbability"])).to(
            torch.device(self.device))
        advantage = torch.from_numpy(np.array(sample["advantage"])).to(
            torch.device(self.device))
        returns = torch.from_numpy(np.array(sample["returns"])).to(
            torch.device(self.device))

        for _ in range(self.ppoEpoch):
            # Hidden states
            actorHiddenStates = (actorHidden[0], actorCell[0])
            criticHiddenStates = (criticHidden[0], criticCell[0])

            values, actionLogProbability, entropy, newActorHidden, newCriticHidden = self.policy.evaluateActions(
                currentAgentObservation, globalObservations, actions, actorHiddenStates, criticHiddenStates
            )

            '''
            ==== ACTOR LOSS FUNCTION ====
            L(theta) = min(importantWeight * A, clip(importantWeight, 1 - ε, 1 + ε) * A) + entropy * σ
            importantWeight = exp(actionLogProbability - oldActionLogProbabilities)

            The Surprising Effectiveness of PPO in Cooperative
            Multi-Agent Games (Page. 11)
            =============================
            advantage: A
            self.entropyCoefficient: σ
            '''

            # Calculate Actor policy loss
            importantWeight = torch.exp(actionLogProbability - oldPolicyActionLogProbability)
            surrogate1 = importantWeight * advantage
            surrogate2 = torch.clamp(importantWeight, 1.0 - self.clipParam, 1.0 + self.clipParam) * advantage
            actorLoss = -torch.mean(torch.min(surrogate1, surrogate2)) - self.entropyCoefficient * entropy

            # Calculate Critic value loss
            criticValueLoss = self.calculateCriticValueLoss(values, valuePredict, returns)

            # Total loss = policy loss + value loss coefficient * value loss
            totalLoss = actorLoss + self.valueLossCoefficient * criticValueLoss

            # Update policy
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            totalLoss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.maximumGradientNorm)
            self.policy.actor_optimizer.step()
            self.policy.critic_optimizer.step()

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
    def train(self, buffer, chunkSize):
        # Generate mini-batch data
        dataBatch = buffer.sampleMiniBatch(self.numMiniBatch, chunkSize)
        for sample in dataBatch:
            self.policyUpdate(sample)