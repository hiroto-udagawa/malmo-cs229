import tensorflow as tf
from collections import deque
import numpy as np
import random
from DeepAgent import DeepAgent
import cv2

image_dim = 84
sess = tf.InteractiveSession()
PIXELS = image_dim * image_dim
neurons = 512
filter1_dim = 8
filter1_depth = 32
filter1_stride = 4
filter2_dim = 4
filter2_depth = 64
filter2_stride = 2
filter3_dim = 3
filter3_depth = 64
filter3_stride = 1

ACTIONS = 9
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000 # timesteps to observe before training
#OBSERVE = 300 # timesteps to observe before training

#OBSERVE = 32 # timesteps to observe before training
EXPLORE = 80000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
#INITIAL_EPSILON = 0.077 # starting value of epsilon
REPLAY_MEMORY = 20000 # number of previous transitions to remember
BATCH = 32 # size of minibatch

FRAMES= 3

class DeepLearner:
    
    def __init__(self):
        self.save = True;
        self.sess = tf.InteractiveSession()      
        self.agent = DeepAgent()
        self.D = deque()
        self.Holdout = deque()
        
        self.s, self.readout, h_fc1  = self.createNet();
        self.s_t = self.a_t = None
        self.epsilon = INITIAL_EPSILON
        self.t = 0
        self.saver = None
        
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.mul(self.readout, self.a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.y - readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        #self.train_step = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.9, momentum=0.95).minimize(cost)
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
          
    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.01)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    #def max_pool_2x2(self, x):
      #return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def createNet(self):
        W_conv1 = self.weight_variable([filter1_dim, filter1_dim, FRAMES, filter1_depth]) #8,8,4,32
        b_conv1 = self.bias_variable([filter1_depth]) #32

        W_conv2 = self.weight_variable([filter2_dim, filter2_dim, filter1_depth, filter2_depth]) #4,4,32,64
        b_conv2 = self.bias_variable([filter2_depth]) #64

        W_conv3 = self.weight_variable([filter3_dim, filter3_dim, filter2_depth, filter3_depth]) #3,3,64,64
        b_conv3 = self.bias_variable([filter3_depth]) #64

        W_fc1 = self.weight_variable([7*7*filter3_depth, neurons]) #7*7*64, 512
        b_fc1 = self.bias_variable([neurons]) #512

        W_fc2 = self.weight_variable([neurons, ACTIONS]) #512
        b_fc2 = self.bias_variable([ACTIONS])

        # input layer
        s = tf.placeholder(tf.float32, [None, image_dim, image_dim, FRAMES])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, filter1_stride) + b_conv1) #stride 4
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, filter2_stride) + b_conv2) #stride 2
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, filter3_stride) + b_conv3) #stride 1
        h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*filter3_depth])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2
        return s, readout, h_fc1

    def initNetwork(self,frame, ob, eval):
        # printing
        #a_file = open("logs/readout.txt", 'w')
        #h_file = open("logs/hidden.txt", 'w')

        x_t = self.agent.resize(self.agent.getPixels(frame))
        #x_t = self.agent.threshold(x_t)
        x_t = x_t.reshape(image_dim, image_dim)
        
        r_0 = self.agent.getReward(ob)
        #terminal = ob[u'IsAlive']    
        terminal = False 
        
        self.s_t = np.stack((x_t, x_t, x_t), axis=2)

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
        x_t1 = x_t1.reshape(image_dim, image_dim ,1)
        
        r_t = self.agent.getReward(ob)        
        s_t1 = np.append(x_t1, self.s_t[:, :, :FRAMES-1], axis=2)
        #cv2.imwrite('messigray.png',x_t1)
        
        #cv2.imwrite('messigray1.png', np.reshape(s_t1[:,:,0], (84,84)))
        #cv2.imwrite('messigray2.png',np.reshape(s_t1[:,:,1], (84,84)))
        #cv2.imwrite('messigray3.png',np.reshape(s_t1[:,:,2], (84,84)))
        # store the transition in D
        
        if self.t < 2000:
            self.Holdout.append((s_t1))
        if self.t % 1000 == 0 and self.t >= 2000:
            readout_batch = self.readout.eval(feed_dict = {self.s : list(self.Holdout)})
            readout_batch = np.array(readout_batch)
            print np.mean(np.amax(readout_batch, axis=1))
            file = open("qvalue.txt", "a")
            file.write(str(np.mean(np.amax(readout_batch, axis=1)))+ "\n")
            file.close()

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
   
        
