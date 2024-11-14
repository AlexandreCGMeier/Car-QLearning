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
print(state_size)
action_size = game.no_of_actions  # 9 possible actions
learning_rate = 0.0025  # Alpha (aka learning rate)

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
starting_episode = 306481
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


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        backgroundColor = [10, 0, 0, 255]
        backgroundColor = [i / 255 for i in backgroundColor]
        self.firstClick = True
        glClearColor(*backgroundColor)
        self.sess = tf.compat.v1.Session()
        self.sess.saver = tf.compat.v1.train.Saver()
        game.new_episode()
        self.state = game.get_state()
        self.nextState = []
        self.episodereward = []

    def on_key_press(self, symbol, modifiers):
        if symbol == key.RIGHT:
            game.car.turningRight = True
        if symbol == key.LEFT:
            game.car.turningLeft = True
        if symbol == key.UP:
            game.car.accelerating = True
        if symbol == key.DOWN:
            game.car.reversing = True

    def on_key_release(self, symbol, modifiers):
        if symbol == key.RIGHT:
            game.car.turningRight = False
        if symbol == key.LEFT:
            game.car.turningLeft = False
        if symbol == key.UP:
            game.car.accelerating = False
        if symbol == key.DOWN:
            game.car.reversing = False

    def on_mouse_press(self, x, y, button, modifiers):
        # print(x,y)
        mode = "RewardGate"
        if self.firstClick:
            self.clickPos = [x, y]
            self.firstClick = not self.firstClick
        else:
            if mode != "RewardGate":
                print("self.walls.append(Wall({}, {}, {}, {}))".format(self.clickPos[0],
                                                                    displayHeight - self.clickPos[1],
                                                                    x, displayHeight - y))
            else:
                print("self.gates.append(RewardGate({}, {}, {}, {}))".format(self.clickPos[0],
                                                                    self.clickPos[1],
                                                                    x, y))
            self.firstClick = not self.firstClick
            # self.gates.append(RewardGate(self.clickPos[0], self.clickPos[1], x, y))

    def on_draw(self):
        game.render()

    def update(self, dt):
        if not game.car.dead:
            game.car.lifespan+=1
            game.car.move()
            game.car.updateControls()
            if game.car.hitAWall():
                game.car.dead = True
                # return
            game.car.checkRewardGates()
            #print(game.car.reward)
            #print(game.car.vel)
        game.car.setVisionVectors()
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