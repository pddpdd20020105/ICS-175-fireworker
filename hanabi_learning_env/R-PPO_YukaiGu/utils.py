import numpy as np

def convertObservationByPlayer(observation, playerID=0):
    return observation['player_observations'][playerID]['vectorized']

def convertObservation(observation):
    playerObservations = observation.get('player_observations', [])
    vectorizedList = []
    for player in playerObservations:
        vector = player.get('vectorized', [])
        vectorizedList.append(np.array(vector, dtype = np.float32))
    if vectorizedList:
        return np.mean(np.stack(vectorizedList), axis = 0)
    else:
        return np.array([], dtype = np.float32)

class Buffer:
    def __init__(self, numAgents):
        # Insert data
        self.numAgents = numAgents
        self.globalObservations = [[] for _ in range(numAgents)]
        self.currentAgentObservation = [[] for _ in range(numAgents)]
        self.actorHidden = [[] for _ in range(numAgents)]
        self.actorCell = [[] for _ in range(numAgents)]
        self.criticHidden = [[] for _ in range(numAgents)]
        self.criticCell = [[] for _ in range(numAgents)]
        self.actions = [[] for _ in range(numAgents)]
        self.rewards = [[] for _ in range(numAgents)]
        self.nextGlobalObservations = [[] for _ in range(numAgents)]
        self.nextAgentObservations = [[] for _ in range(numAgents)]
        # Calculated data
        self.valuePredict = [[] for _ in range(numAgents)]
        self.returns = [[] for _ in range(numAgents)]
        self.advantage = [[] for _ in range(numAgents)]
        self.oldPolicyActionLogProbability = [[] for _ in range(numAgents)]

        # PopArt normalization
        self.advantagePopartMean = [0.0 for _ in range(self.numAgents)]
        self.advantagePopartVar = [1.0 for _ in range(self.numAgents)]
        self.returnPopartMean = [0.0 for _ in range(self.numAgents)]
        self.returnPopartVar = [1.0 for _ in range(self.numAgents)]

    def insert(self, agentID, globalObservations, currentAgentObservation, actorHidden, actorCell,
               criticHidden, criticCell, actions, rewards, nextGlobalObservations, nextAgentObservations,
               valuePredict, oldPolicyActionLogProbability):
        globalObservationsVectorized = convertObservation(globalObservations)
        self.globalObservations[agentID].append(globalObservationsVectorized)

        currentAgentObservationVectorized = currentAgentObservation['vectorized']
        self.currentAgentObservation[agentID].append(currentAgentObservationVectorized)

        self.actorHidden[agentID].append(actorHidden)
        self.actorCell[agentID].append(actorCell)
        self.criticHidden[agentID].append(criticHidden)
        self.criticCell[agentID].append(criticCell)
        self.actions[agentID].append(actions)
        self.rewards[agentID].append(rewards)
        convertedNextGlobalObservation = convertObservation(nextGlobalObservations)
        self.nextGlobalObservations[agentID].append(convertedNextGlobalObservation)

        convertedNextAgentObservation = convertObservation(nextAgentObservations)
        self.nextAgentObservations[agentID].append(convertedNextAgentObservation)

        self.valuePredict[agentID].append(valuePredict)
        self.oldPolicyActionLogProbability[agentID].append(oldPolicyActionLogProbability)

    def clear(self):
        self.globalObservations = [[] for _ in range(self.numAgents)]
        self.currentAgentObservation = [[] for _ in range(self.numAgents)]
        self.actorHidden = [[] for _ in range(self.numAgents)]
        self.actorCell = [[] for _ in range(self.numAgents)]
        self.criticHidden = [[] for _ in range(self.numAgents)]
        self.criticCell = [[] for _ in range(self.numAgents)]
        self.actions = [[] for _ in range(self.numAgents)]
        self.rewards = [[] for _ in range(self.numAgents)]
        self.nextGlobalObservations = [[] for _ in range(self.numAgents)]
        self.nextAgentObservations = [[] for _ in range(self.numAgents)]

        self.valuePredict = [[] for _ in range(self.numAgents)]
        self.returns = [[] for _ in range(self.numAgents)]
        self.advantage = [[] for _ in range(self.numAgents)]
        self.oldPolicyActionLogProbability = [[] for _ in range(self.numAgents)]

    def computeReturnsAndAdvantages(self, model, gamma=0.99, lamda=0.95, beta=0.001, epsilon=1e-8):
        for agentID in range(self.numAgents):
            rewards = np.array(self.rewards[agentID])
            valuePredict = np.array(self.valuePredict[agentID])
            T = rewards.shape[0]
            returns = np.zeros_like(rewards)
            advantage = np.zeros_like(rewards)
            lastGAE = 0
            for t in reversed(range(T)):
                # If is last step
                if t == T - 1:
                    delta = rewards[t] - valuePredict[t]
                else:
                    '''
                    ======== DELTA Equation ======
                    δt = r_t + γ * V(s_(t+1) − V(s_t)
                    
                    HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED 
                    ADVANTAGE ESTIMATION (Fig. 17)
                    ==============================
                    r_t: reward at time t
                    V(s_t): value at s_t
                    V(s_(t+1)): next value at s_(t+1)
                    γ: gamma, the discount factor
                    '''
                    nextValue = valuePredict[t + 1]
                    delta = rewards[t] + gamma * nextValue - valuePredict[t]
                '''
                ====== ADVANTAGE Equation ======
                A_t = δt + γ * λ * A_(t+1)
                
                HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED 
                ADVANTAGE ESTIMATION (Fig. 16)
                ================================
                γ: gamma, the discount factor
                λ: lamda
                '''
                lastGAE = delta + gamma * lamda * lastGAE
                advantage[t] = lastGAE
                returns[t] = advantage[t] + valuePredict[t]

            # Normalize return
            oldReturnMean = self.returnPopartMean[agentID]
            oldReturnVar = self.returnPopartVar[agentID]
            self.returnPopartMean[agentID] = (1 - beta) * self.returnPopartMean[agentID] + beta * returns.mean()
            self.returnPopartVar[agentID] = (1 - beta) * self.returnPopartVar[agentID] + beta * np.var(returns)
            sigma = np.sqrt(self.returnPopartVar[agentID])
            normalizedReturns = (returns - self.returnPopartMean[agentID]) / (sigma + epsilon)
            model.popartUpdateCritic(oldReturnMean, oldReturnVar, self.returnPopartMean[agentID], self.returnPopartVar[agentID])

            # Normalize advantage
            self.advantagePopartMean[agentID] = (1 - beta) * self.advantagePopartMean[agentID] + beta * advantage.mean()
            self.advantagePopartVar[agentID] = (1 - beta) * self.advantagePopartVar[agentID] + beta * np.var(advantage)
            sigma = np.sqrt(self.advantagePopartVar[agentID])
            normalizedAdvantage = (advantage - self.advantagePopartMean[agentID]) / (sigma + epsilon)

            self.returns[agentID] = normalizedReturns.tolist()
            self.advantage[agentID] = normalizedAdvantage.tolist()

    def sampleMiniBatch(self, numMiniBatch, chunkLength):
        T = min(len(self.globalObservations[i]) for i in range(self.numAgents))
        N = self.numAgents
        batchSize = T * N

        globalObservationsArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.globalObservations[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            globalObservationsArr.append(row)
        globalObservationsArr = np.stack(globalObservationsArr, axis = 0)

        currentAgentObsArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.currentAgentObservation[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            currentAgentObsArr.append(row)
        currentAgentObsArr = np.stack(currentAgentObsArr, axis = 0)

        actorHiddenArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.actorHidden[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            actorHiddenArr.append(row)
        actorHiddenArr = np.stack(actorHiddenArr, axis = 0)

        actorCellArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.actorCell[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            actorCellArr.append(row)
        actorCellArr = np.stack(actorCellArr, axis = 0)

        criticHiddenArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.criticHidden[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            criticHiddenArr.append(row)
        criticHiddenArr = np.stack(criticHiddenArr, axis = 0)

        criticCellArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.criticCell[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            criticCellArr.append(row)
        criticCellArr = np.stack(criticCellArr, axis = 0)

        actionsArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.actions[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            actionsArr.append(row)
        actionsArr = np.stack(actionsArr, axis = 0)

        rewardsArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(self.rewards[i][t])
            row = np.array(row, dtype = np.float32)
            rewardsArr.append(row)
        rewardsArr = np.stack(rewardsArr, axis = 0)

        nextGlobalObsArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(np.array(self.nextGlobalObservations[i][t], dtype = np.float32))
            row = np.stack(row, axis = 0)
            nextGlobalObsArr.append(row)
        nextGlobalObsArr = np.stack(nextGlobalObsArr, axis = 0)

        valuePredictArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(self.valuePredict[i][t])
            row = np.array(row, dtype = np.float32)
            valuePredictArr.append(row)
        valuePredictArr = np.stack(valuePredictArr, axis = 0)  # shape (T, N)

        oldPolicyActionLogProbArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(self.oldPolicyActionLogProbability[i][t])
            row = np.array(row, dtype = np.float32)
            oldPolicyActionLogProbArr.append(row)
        oldPolicyActionLogProbArr = np.stack(oldPolicyActionLogProbArr, axis = 0)

        advantageArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(self.advantage[i][t])
            row = np.array(row, dtype = np.float32)
            advantageArr.append(row)
        advantageArr = np.stack(advantageArr, axis = 0)

        returnsArr = []
        for t in range(T):
            row = []
            for i in range(N):
                row.append(self.returns[i][t])
            row = np.array(row, dtype = np.float32)
            returnsArr.append(row)
        returnsArr = np.stack(returnsArr, axis = 0)

        dataChunks = (T * N) // chunkLength
        mini_batch_size = dataChunks // numMiniBatch

        perm = np.random.permutation(dataChunks)
        sampler = [perm[i * mini_batch_size: (i + 1) * mini_batch_size] for i in range(numMiniBatch)]

        def flatten(arr):
            return arr.reshape(T * N, *arr.shape[2:])

        def chunkify(arr):
            total = arr.shape[0]
            numChunks = total // chunkLength
            trimmed = arr[:numChunks * chunkLength]
            return trimmed.reshape(numChunks, chunkLength, *arr.shape[1:])

        globalObservationsChunk = chunkify(flatten(globalObservationsArr))
        currentAgentObservationChunk = chunkify(flatten(currentAgentObsArr))
        actorHiddenChunk = chunkify(flatten(actorHiddenArr))
        actorCellChunk = chunkify(flatten(actorCellArr))
        criticHiddenChunk = chunkify(flatten(criticHiddenArr))
        criticCellChunk = chunkify(flatten(criticCellArr))
        actionsChunk = chunkify(flatten(actionsArr))
        rewardsChunk = chunkify(flatten(rewardsArr[..., np.newaxis]))
        nextGlobalObservationsChunk = chunkify(flatten(nextGlobalObsArr))
        valuePredictChunk = chunkify(flatten(valuePredictArr[..., np.newaxis]))
        oldPolicyActionLogProbabilityChunk = chunkify(flatten(oldPolicyActionLogProbArr[..., np.newaxis]))
        advantageChunk = chunkify(flatten(advantageArr[..., np.newaxis]))
        returnsChunk = chunkify(flatten(returnsArr[..., np.newaxis]))

        for indices in sampler:
            globalObservations = globalObservationsChunk[indices].reshape(-1, *globalObservationsChunk.shape[2:])
            currentAgentObservation = currentAgentObservationChunk[indices].reshape(-1, *currentAgentObservationChunk.shape[2:])
            actorHidden = actorHiddenChunk[indices].reshape(-1, *actorHiddenChunk.shape[2:])
            actorCell = actorCellChunk[indices].reshape(-1, *actorCellChunk.shape[2:])
            criticHidden = criticHiddenChunk[indices].reshape(-1, *criticHiddenChunk.shape[2:])
            criticCell = criticCellChunk[indices].reshape(-1, *criticCellChunk.shape[2:])
            actions = actionsChunk[indices].reshape(-1, *actionsChunk.shape[2:])
            rewards = rewardsChunk[indices].reshape(-1, *rewardsChunk.shape[2:])
            nextGlobalObservations = nextGlobalObservationsChunk[indices].reshape(-1, *nextGlobalObservationsChunk.shape[2:])
            valuePredict = valuePredictChunk[indices].reshape(-1, *valuePredictChunk.shape[2:])
            oldPolicyActionLogProbability = oldPolicyActionLogProbabilityChunk[indices].reshape(-1, *oldPolicyActionLogProbabilityChunk.shape[2:])
            advantage = advantageChunk[indices].reshape(-1, *advantageChunk.shape[2:])
            returns = returnsChunk[indices].reshape(-1, *returnsChunk.shape[2:])

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