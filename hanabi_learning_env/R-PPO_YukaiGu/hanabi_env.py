from hanabi_learning_environment.rl_env import HanabiEnv
from hanabi_learning_environment import pyhanabi

class CustomHanabiEnv(HanabiEnv):
    def __init__(self, config):
        super(CustomHanabiEnv, self).__init__(config)
        self.rewardMap = {
            # PLAY ACTION
            "playWrongCard1":       -0.1,
            "playWrongCard2":       -0.5,
            "playWrongCard3":       -1,

            # HINT ACTION
            "hintColor":            +0.05,
            "hintRank":             +0.05,
            "hintWithNoToken":      -0.01,

            # DISCARD ACTION
            "discardToGainToken":   +0.01,
            "discardWhenHasToken":  -0.02,

            # OTHER
            "notDone":              0,
        }
        self.playWrongCardCount = 0

    def step(self, action):
        if isinstance(action, dict):
            # Convert dict action HanabiMove
            action = self._build_move(action)
        elif isinstance(action, int):
            # Convert int action into a Hanabi move.
            action = self.game.get_move(action)
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(
                action))

        last_score = self.state.score()

        # Apply action
        self.state.apply_move(action)

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        observation = self._make_observation_all_players()
        done = self.state.is_terminal()
        info = {}

        # Card 1 = 2 reward, Card 2 = 4 reward, Card 3 = 6 reward, Card 4 = 8 reward, Card 5 = 10 reward
        baseReward = (self.state.score() - last_score) * 2

        # Giving additional reward based on the action
        addingReward = 0
        actionType = action.to_dict()["action_type"]
        if actionType == "PLAY":
            # If play the wrong card
            if self.state.score() == last_score:
                self.playWrongCardCount += 1
                # First wrong
                if self.playWrongCardCount == 1:
                    addingReward += self.rewardMap["playWrongCard1"]
                # Second wrong
                elif self.playWrongCardCount == 2:
                    addingReward += self.rewardMap["playWrongCard2"]
                # Third wrong
                else:
                    addingReward += self.rewardMap["playWrongCard3"]
        elif actionType == "DISCARD":
            tokens = self.state.information_tokens()
            # Discard to gain token
            if tokens == 0:
                addingReward += self.rewardMap["discardToGainToken"]
            # Discard when has token
            elif tokens > 4:
                addingReward += self.rewardMap["discardWhenHasToken"]
        elif actionType == "REVEAL_COLOR":
            addingReward += self.rewardMap["hintColor"]
            if self.state.information_tokens() == 0:
                addingReward += self.rewardMap["hintWithNoToken"]
        elif actionType == "REVEAL_RANK":
            addingReward += self.rewardMap["hintRank"]
            if self.state.information_tokens() == 0:
                addingReward += self.rewardMap["hintWithNoToken"]

        if not done:
            addingReward += self.rewardMap["notDone"]

        reward = baseReward + addingReward

        return observation, reward, done, info