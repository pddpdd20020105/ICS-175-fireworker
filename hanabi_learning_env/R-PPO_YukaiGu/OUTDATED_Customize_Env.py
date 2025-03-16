import gym
import hanabi_learning_environment.rl_env as rl_env
import random
from collections import Counter
from utils import convertObservation, convertObservationByPlayer

class HanabiEnv(gym.Env):
    def __init__(self):
        self.rewardMap = {
            # ACTION CHOICE
            "illegalMove":          -5,
            "illegalMoveIncrease":  -0.5,
            "legalMove":            +2,
            "noLegalMove":          -5,
            # PLAY CARD
            "playWrongCard1":       -8,
            "playWrongCard2":       -15,
            "playWrongCard3":       -30,
            "playCorrectCard1":     +10,
            "playCorrectCard2":     +15,
            "playCorrectCard3":     +22,
            "playCorrectCard4":     +30,
            "playCorrectCard5":     +40,
            # HINT CARD
            "hintColor":            +0,
            "hintRank":             +0,
            "hintWithNoToken":      -3,
            "hintColorGood":        +3,
            "hintColorBad":         -3,
            "hintRankGood":         +3,
            "hintRankBad":          -3,
            # DISCARD CARD
            "discardToGainToken":   +2,
            "discardWhenFullToken": -3,
            "discardUniqueCard":    -10,
            # OTHER
            "notDone": +0.5,
        }
        self.noVariantConfig = {
            "players": 2,
            "random_start_player": True,
            "colors": 5,
            "ranks": 5,
            "hand_size": 5,
            "max_information_tokens": 8,
            "max_life_tokens": 3
        }
        self.playWrongCardCount = 0
        self.illegalMoveCount = 0
        self.env = rl_env.HanabiEnv(self.noVariantConfig)
        self.actionSpace = gym.spaces.Discrete(self.env.num_moves())
        sampleObservation = self.env.reset()
        vectorizedSize = len(convertObservation(sampleObservation))
        self.observationSpace = gym.spaces.Box(low = 0, high = 1, shape = (vectorizedSize,))

    def getCurrentPlayer(self):
        return self.env.state.cur_player()

    def getGameScore(self):
        return sum(self.env.state.fireworks())

    def reset(self):
        observation = self.env.reset()
        return convertObservation(observation), {}

    def isUniqueCard(self, discardCard):
        rank = discardCard["rank"]
        color = discardCard["color"]

        discardCounts = Counter((card.color, card.rank) for card in self.env.state.discard_pile())
        cardTotal = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}

        return discardCounts[(color, rank)] >= cardTotal[rank]

    def calculateReward(self, actionType, playedCard, discardCard, beforeToken, fireworksBefore,
                        fireworksAfter, log, done, addingRewards):
        addingReward = addingRewards

        if actionType == "PLAY":
            # If play the wrong card
            if fireworksBefore == fireworksAfter:
                if log:
                    print("\033[1;31m PLAY WRONG CARD\033[0m")
                self.playWrongCardCount += 1
                if self.playWrongCardCount == 1:
                    addingReward += self.rewardMap["playWrongCard1"]
                elif self.playWrongCardCount == 2:
                    addingReward += self.rewardMap["playWrongCard2"]
                elif self.playWrongCardCount >= 3:
                    addingReward += self.rewardMap["playWrongCard3"]
            # If play the correct card
            else:
                if log:
                    print("\033[1;32m PLAY CORRECT CARD\033[0m")
                rank = playedCard['rank'] + 1
                addingReward += self.rewardMap["playCorrectCard" + str(rank)]
        elif actionType == "DISCARD":
            if beforeToken == 0:
                addingReward += self.rewardMap["discardToGainToken"]
            elif beforeToken == self.noVariantConfig["max_information_tokens"]:
                addingReward += self.rewardMap["discardWhenFullToken"]
        elif actionType == "REVEAL_COLOR":
            addingReward += self.rewardMap["hintColor"]
            if beforeToken == 0:
                addingReward += self.rewardMap["hintWithNoToken"]
        elif actionType == "REVEAL_RANK":
            addingReward += self.rewardMap["hintRank"]

            if beforeToken == 0:
                addingReward += self.rewardMap["hintWithNoToken"]

        if not done:
            addingReward += self.rewardMap["notDone"]

        return addingReward

    def step(self, actionIndex, log=False):
        currentPlayer = self.env.state.cur_player()
        observation = self.env.state.observation(currentPlayer)
        legalAction = observation.legal_moves()
        action = self.env.game.get_move(actionIndex)
        addingReward = 0

        fireworksBefore = observation.fireworks().copy()
        if log:
            print(f"Fireworks Before: {fireworksBefore}")
            print(action)

        if not legalAction:
            print("\033[1;33m <WARNING: NO LEGAL MOVES> \033[0m")
            return convertObservation(self.env.state.observation(currentPlayer)), self.rewardMap["noLegalMove"], False, {}

        # Illegal Move
        action = action.to_dict()
        if any(action == a.to_dict() for a in legalAction):
            addingReward +=  self.rewardMap["legalMove"]
        else:
            if log:
                print(f"\033[1;33m ILLEGAL MOVE!\033[0m")
            self.illegalMoveCount += 1
            addingReward += self.rewardMap["illegalMove"] + self.illegalMoveCount * self.rewardMap["illegalMoveIncrease"]
            # If choose illegal move, choose a random discard move, if possible
            discardActions = [a for a in legalAction if a.to_dict()["action_type"] == "DISCARD"]
            if discardActions:
                action = random.choice(discardActions).to_dict()
            else:
                action = random.choice(legalAction).to_dict()

        # Get action description
        actionType = action["action_type"]
        playedCard = None
        discardCard = None
        beforeToken = self.env.state.information_tokens()
        if actionType == "PLAY":
            playedCard = self.env.state.player_hands()[currentPlayer][action["card_index"]].to_dict()
            actionDescription = f"PLAY CARD {playedCard['color']} {playedCard['rank'] + 1}"
        elif actionType == "DISCARD":
            discardCard = self.env.state.player_hands()[currentPlayer][action["card_index"]].to_dict()
            actionDescription = f"DISCARD CARD {discardCard['color']} {discardCard['rank'] + 1}"
        elif actionType == "REVEAL_COLOR":
            actionDescription = f"REVEAL COLOR {action['color']} TO PLAYER {action['target_offset'] + 1}"
        elif actionType == "REVEAL_RANK":
            actionDescription = f"REVEAL NUMBER {action['rank'] + 1} TO PLAYER {action['target_offset'] + 1}"
        else:
            actionDescription = f"UNKNOWN ACTION {action}"

        ######################## Do the action ########################
        obs, reward, done, info = self.env.step(action)
        fireworksAfter = self.env.state.observation(currentPlayer).fireworks().copy()
        ######################## Do the action ########################

        # Calculate reward
        reward += self.calculateReward(actionType, playedCard, discardCard, beforeToken,
                                       fireworksBefore, fireworksAfter, log, done, addingReward)

        # Outlog the action and reward
        if log:
            print(f"Reward: \033[1m {str(reward).rjust(2)} \033[0m, |"
                  f" Action: \033[1m {actionDescription.ljust(30)} \033[0m")
            print(f"After fireworks: {fireworksAfter}")
            print()

        return convertObservation(obs), reward, done, info