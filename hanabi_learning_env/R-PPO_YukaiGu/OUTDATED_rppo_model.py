import torch
import torch.nn as nn
import torch.nn.functional as F
from hanabi_env import HanabiEnv

class RPPOModel(nn.Module):
    def __init__(self, inputDimension, hiddenDimension, actionDimension):
        super(RPPOModel, self).__init__()
        self.lstm = nn.LSTM(inputDimension, hiddenDimension, batch_first=True)
        self.actor = nn.Linear(hiddenDimension, actionDimension)
        self.critic = nn.Linear(hiddenDimension, 1)

    def forward(self, x, hiddenState):
        x, hiddenState = self.lstm(x, hiddenState)
        actionProbabilities = F.softmax(self.actor(x), dim = -1)
        statusValue = self.critic(x)
        return actionProbabilities, statusValue, hiddenState

def createRPPOModel():
    env = HanabiEnv()
    observation, _ = env.reset()
    inputDimension = len(observation)
    hiddenDimension = 128
    actionDimension = env.actionSpace.n
    model = RPPOModel(inputDimension, hiddenDimension, actionDimension)
    return model

def saveModel(model, filename="rppo_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def loadModel(filename="rppo_model.pth"):
    model = createRPPOModel()
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return model

def loadModel(filename="rppo_model.pth"):
    model = createRPPOModel()
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model loaded from {filename}")
    return model