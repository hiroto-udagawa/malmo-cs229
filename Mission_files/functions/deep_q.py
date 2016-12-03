import tensorflow as tf
from collections import deque
import numpy as np
import random
from DeepAgent import DeepAgent

# image_width = 480
# image_height = 640
image_width = 80
image_height = 80
sess = tf.InteractiveSession()
PIXELS = image_width * image_height
neurons = 1024
first_layer_filter = 32
layer_size = 5
second_layer_filter = 64

ACTIONS = 4
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
#OBSERVE = 32. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION=1
FRAMES= 4

class DeepLearner:
    
    def __init__(self):
        self.sess = tf.InteractiveSession()      
        self.agent = DeepAgent()
        self.D = deque()
        
        self.s, self.readout, h_fc1  = self.createNet();
        self.train_step = self.s_t = self.y = self.a = self.a_t = None
        self.epsilon = INITIAL_EPSILON
        self.t = 0
          
    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def createNet(self):
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, ACTIONS])
        b_fc2 = self.bias_variable([ACTIONS])

        # input layer
        s = tf.placeholder("float", [None, image_height, image_width, FRAMES])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        return s, readout, h_fc1

    def initNetwork(self,frame, ob):
        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.mul(self.readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # open up a game state to communicate with emulator
        # store the previous observations in replay memory

        # printing
        a_file = open("logs/readout.txt", 'w')
        h_file = open("logs/hidden.txt", 'w')

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        #x_t = frame
        x_t = self.agent.resize( self.agent.getPixels(frame))
        r_0 = self.agent.getReward(ob)
        terminal = ob[u'IsAlive']    
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        # saving and loading networks
        saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        
        self.a = a
        self.y = y
        self.train_step = train_step
        self.s_t = s_t
        
        readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= self.epsilon:
        #if True:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        self.a_t = a_t
        return action_index

    
    def trainNetwork(self, frame, ob):
        # scale down epsilon
        if self.epsilon > FINAL_EPSILON and self.t > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        #x_t1 = np.reshape(frame, ( 640,480,1) )
        x_t1 = self.agent.resize( self.agent.getPixels(frame))
        x_t1 = x_t1.reshape(80,80,1)
        
        r_t = self.agent.getReward(ob)
        terminal = ob[u'IsAlive']
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)

        s_t1 = np.append(x_t1, self.s_t[:, :, :3], axis=2)

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
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        self.a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= self.epsilon:
            print("----------Random Action----------")
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

        print("TIMESTEP", self.t, "/ STATE", state, \
            "/ EPSILON", self.epsilon, "/ ACTION", self.a_t, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''
        
        return action_index
