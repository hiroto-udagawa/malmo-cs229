import json
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
class DeepAgent:
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.cum_reward = 0

        #self.actions = ["move 1", "move 0", "move -1", "turn 0.7", "turn -0.7"]      
        #self.antiActions = ["move 0", "move 0", "move 0", "turn 0", "turn 0"]
        
        self.actions = ["move 1", "move 0", "move -1", "turn 0.9", "turn -0.9", "turn 0.5", "turn -0.5", "turn 0.1", "turn -0.1"]      
        self.antiActions = ["move 0", "move 0", "move 0", "turn 0", "turn 0", "turn 0", "turn 0", "turn 0", "turn 0"]
        

        self.rewards = {"health":-5 , "kills":40, "time":0.03, "hit":2.5}
        self.currentHealth = 20
        self.kills = 0
        
    def getReward(self, ob):
        reward = 0
        if ('MobsKilled' not in ob) or ('LineOfSight' not in ob):
            return 0      
        reward += (ob[u'MobsKilled'] - self.kills) * self.rewards['kills']
        reward += (self.currentHealth - ob[u'Life']) * self.rewards["health"]
        reward += self.rewards["time"]
        if ob[u'LineOfSight'][u'hitType'] == 'entity' and ob[u'LineOfSight'][u'inRange'] == True:
            reward += self.rewards["hit"]
        
        #print self.currentHealth, "    ", ob[u'Life']
        self.currentHealth = ob[u'Life']
        self.kills = ob[u'MobsKilled']
        self.cum_reward += reward
        return reward

    def getPixels(self, frame):                                    
        '''
        Retrieves pixels from the frame object
        '''
        width = frame.width                                
        height = frame.height                              
        channels = frame.channels                          
        pixels = np.array(frame.pixels, dtype = np.uint8)       
        img = np.reshape(pixels, (height, width, channels))                        
        return img      
                                                           
    def resize(self, image):

        #return cv2.resize(image,(80,80))[:,:,2]
        return cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_RGB2GRAY)
	
    def threshold(self, image):
        #retval, th_image = cv2.threshold(image,1,255,cv2.THRESH_BINARY) 
        retval, th_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th_image
