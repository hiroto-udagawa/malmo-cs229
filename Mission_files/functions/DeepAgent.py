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
        #self.actions = ["move 1", "move 0", "move -1", "turn -1", "turn 1", "turn 0", "attack 1", "attack 0"]
        self.actions = ["move 1",  "move -1", "turn -0.8", "turn 0.8", "move 0"]  
        self.antiActions = ["move 0", "move 0", "turn 0", "turn 0", "move 0"]
        self.rewards = {"health":-5 , "kills":100, "time":0.02, "hit":0.5}
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
        '''
            Resizes the image to 80 by 80, works only if the dimensions are
            multiples of 80
            '''         
    	'''                        
            dim1 = image.shape[0]                              
            dim2 = image.shape[1]                              
            stride1 = dim1 / 80                                
            stride2 = dim2 / 80                                
            return image[::stride1, ::stride2]
    	'''
        return cv2.resize(image,(80,80))[:,:,2]
        #return cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_RGB2GRAY)
	
    def threshold(self, image):
        #retval, th_image = cv2.threshold(image,1,255,cv2.THRESH_BINARY) 
        retval, th_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th_image
