import tensorflow as tf
from collections import deque
import numpy as np
import random
from DeepAgent import DeepAgent
import cv2

image_width = 84
image_height = 84
sess = tf.InteractiveSession()
PIXELS = image_width * image_height
neurons = 1024
first_layer_filter = 32
layer_size = 5
second_layer_filter = 64

ACTIONS = 7
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000 # timesteps to observe before training
#OBSERVE = 32 # timesteps to observe before training
EXPLORE = 50000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
#INITIAL_EPSILON = 0.077 # starting value of epsilon
REPLAY_MEMORY = 10000 # number of previous transitions to remember
BATCH = 32 # size of minibatch

FRAME_PER_ACTION=1
FRAMES= 4

class DeepLearner:
    
    def __init__(self):
        self.sess = tf.InteractiveSession()      
        self.agent = DeepAgent()
        self.D = deque()
        
        self.s, self.readout, h_fc1  = self.createNet();
        self.s_t = self.a_t = None
        self.epsilon = INITIAL_EPSILON
        self.t = 0
        self.saver = None
        
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.mul(self.readout, self.a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.y - readout_action))
        #self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-6, decay=0.9, momentum=0.95).minimize(cost)
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
          
    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    #def max_pool_2x2(self, x):
      #return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def createNet(self):
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([7*7*64, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, ACTIONS])
        b_fc2 = self.bias_variable([ACTIONS])

        # input layer
        s = tf.placeholder(tf.float32, [None, 84, 84, FRAMES])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*64])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2
        return s, readout, h_fc1

    def initNetwork(self,frame, ob, eval):
        # printing
        #a_file = open("logs/readout.txt", 'w')
        #h_file = open("logs/hidden.txt", 'w')

        x_t = self.agent.resize( self.agent.getPixels(frame))
        #x_t = self.agent.threshold(x_t)
        x_t = x_t.reshape(84,84)
        
        r_0 = self.agent.getReward(ob)
        #terminal = ob[u'IsAlive']    
        terminal = False 
        
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        # saving and loading networks

        readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        self.a_t = np.zeros([ACTIONS])
        if not eval and random.random() <= self.epsilon:
        #if True:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            self.a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            self.a_t[action_index] = 1
        return action_index

    
    def trainNetwork(self, frame, ob, terminal):
        # scale down epsilon
        if self.epsilon > FINAL_EPSILON and self.t > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1 = self.agent.resize( self.agent.getPixels(frame))
        #cv2.imwrite('messigray.png',x_t1)
        x_t1 = x_t1.reshape(84,84,1)
        
        r_t = self.agent.getReward(ob)        
        s_t1 = np.append(x_t1, self.s_t[:, :, :3], axis=2)
        #cv2.imwrite('messigray.png',x_t1)
        
        #cv2.imwrite('messigray1.png', np.reshape(s_t1[:,:,0], (84,84)))
        #cv2.imwrite('messigray2.png',np.reshape(s_t1[:,:,1], (84,84)))
        #cv2.imwrite('messigray3.png',np.reshape(s_t1[:,:,2], (84,84)))
        #cv2.imwrite('messigray4.png',np.reshape(s_t1[:,:,3], (84,84)))
        # store the transition in D
        self.D.append((self.s_t, self.a_t, r_t, s_t1, terminal))

        
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()

        # only train if done observing
        if self.t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(self.D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            
            y_batch = []
            readout_j1_batch = self.readout.eval(feed_dict = {self.s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                               
            # perform gradient step
            self.train_step.run(feed_dict = {
                self.y : y_batch,
                self.a : a_batch,
                self.s : s_j_batch}
            )

        # update the old values
        self.s_t = s_t1
        self.t += 1

        # save progress every 10000 iterations
        if self.t % 10000 == 0:
            self.saver.save(self.sess, 'networks/zombie-dqn', global_step = self.t)

        readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        self.a_t = np.zeros([ACTIONS])
        if random.random() <= self.epsilon:
            #print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            self.a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            self.a_t[action_index] = 1
            
        # print info
        state = ""
        if self.t <= OBSERVE:
            state = "observe"
        elif self.t > OBSERVE and self.t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if self.t % 100 == 0:
            print("TIMESTEP", self.t, "/ STATE", state, \
                "/ EPSILON", self.epsilon, "/ ACTION", self.a_t, "/ REWARD", r_t, \
                "/ Q_MAX %e" % np.max(readout_t))
        
        return action_index
        
    def evalNetwork(self, frame, ob):
        
        x_t1 = self.agent.resize( self.agent.getPixels(frame))
        x_t1 = x_t1.reshape(84,84,1)
        r_t = self.agent.getReward(ob)

        terminal = False 
        s_t1 = np.append(x_t1, self.s_t[:, :, :3], axis=2)
        
        self.s_t = s_t1
        self.t += 1

        # save progress every 10000 iterations
        if self.t % 10000 == 0:
            self.saver.save(self.sess, 'networks/zombie-dqn', global_step = self.t)

        readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        self.a_t = np.zeros([ACTIONS])
        action_index = np.argmax(readout_t)
        self.a_t[action_index] = 1
        
        # print info
        state = ""
        if self.t <= OBSERVE:
            state = "observe"
        elif self.t > OBSERVE and self.t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if self.t % 100 == 0:
            print("TIMESTEP", self.t, "/ STATE", state, \
                "/ EPSILON", self.epsilon, "/ ACTION", self.a_t, "/ REWARD", r_t, \
                "/ Q_MAX %e" % np.max(readout_t))
                
        return action_index
            
        
