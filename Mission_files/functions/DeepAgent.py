import json
import logging
import os
import random
import sys
import time

class DeepAgent:
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.01 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["move 1", "move 0", "move -1", "turn -1", "turn 1", "turn 0", "attack 1", "attack 0"]
        self.rewards = {"health":-5 , "kills":20, "time":0.1}
        self.q_table = {}
        self.currentHealth = 20
        self.kills = 0
        
    def getReward(self, ob):
        reward = 0
        reward += (ob[u'MobsKilled'] - self.kills) * self.rewards['kills']
        reward += (self.currentHealth - ob[u'Life']) * self.rewards["health"]
        reward += self.rewards["time"]
               
        self.currentHealth = ob[u'Life']
        self.kills = ob[u'MobsKilled']
        return reward
         

                