import pyglet
from pyglet.gl import *
import pygame
import math
from pyglet.window import key
from Drawer import Drawer
# from PygameAdditionalMethods import *
from ShapeObjects import Line
import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
from collections import deque
import random
import os
from Globals import displayHeight, displayWidth
from Game import Game
from OutsourcedClasses import DDDQNNet, Memory
from pyglet import clock

frameRate = 120.0

vec2 = pygame.math.Vector2
tf.compat.v1.disable_eager_execution()

game = Game()
possible_actions = np.identity(game.no_of_actions, dtype=int).tolist()

### MODEL HYPERPARAMETERS
state_size = [game.state_size]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = game.no_of_actions  # 9 possible actions
learning_rate = 0.1  # Alpha (aka learning rate) # 0.00025

### TRAINING HYPERPARAMETERS
total_episodes = 175000 + 2000  # Total episodes for training
max_steps = 10000  # Max possible steps in an episode
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 10000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 0.3  # exploration probability at start, CHANGED THIS AS WE START WITH ADVANCED MODEL
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001 # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95  # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
memory_size = 100000  # Number of experiences the Memory can keep #100000
pretrain_length = memory_size  # Number of experiences stored in the Memory when initialized for the first time

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training =  False
load = True
starting_episode = 6208
load_traing_model = True
load_training_model_number = starting_episode

# Reset the graph
tf.compat.v1.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")

# Instantiate the target network
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

# Instantiate memory
memory = Memory(memory_size)

# Render the environment
game.new_episode()



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):

    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability

def update_target_graph():
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, "./allModels/model{}/models/model.ckpt".format(load_training_model_number))

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        # set background color
        backgroundColor = [10, 0, 0, 255]
        backgroundColor = [i / 255 for i in backgroundColor]
        self.firstClick = True
        glClearColor(*backgroundColor)
        # load background image
        self.sess = tf.compat.v1.Session()
        self.sess.saver = tf.compat.v1.train.Saver()
        self.load_training_model_number = load_training_model_number
        game.new_episode()
        self.state = game.get_state()
        self.nextState = []
        self.loadSession()
        self.episodereward = []

    def loadSession(self):
        if load_traing_model:
            directory = "./allModels/model{}/models/model.ckpt".format(self.load_training_model_number)
            self.sess.saver.restore(self.sess, directory)

    def on_mouse_press(self, x, y, button, modifiers):
        # print(x,y)
        if self.firstClick:
            self.clickPos = [x, y]
        else:
            print("self.gates.append(RewardGate({}, {}, {}, {}))".format(self.clickPos[0],
                                                                   displayHeight - self.clickPos[1],
                                                                   x, displayHeight - y))
        
            self.gates.append(RewardGate(self.clickPos[0], self.clickPos[1], x, y))

    def on_draw(self):
        game.render()
        #print(clock.get_fps())

    def update(self, dt):
        exp_exp_tradeoff = np.random.rand()
        if load_traing_model:
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * load_training_model_number* 100)
        else:
            explore_probability = 0.0001

        if explore_probability > exp_exp_tradeoff:
        #if True: #As of 21/05/24, frame rate sits at approx. 60fps with random choice on XPS13!
            # Make a random action (exploration)
            action = random.choice(possible_actions)

        else:
            Qs = self.sess.run(DQNetwork.output,
                               feed_dict={DQNetwork.inputs_: self.state.reshape((1, *self.state.shape))})
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

        reward = game.make_action(action)
        
        # game.render()
        done = game.is_episode_finished()

        if done:
            game.new_episode()
            self.state = game.get_state()
        else:
            self.next_state = game.get_state()
            self.state = self.next_state


window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
pyglet.clock.schedule_interval(window.update, 1 / frameRate)
pyglet.app.run()