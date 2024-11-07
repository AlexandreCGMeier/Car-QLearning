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
from tqdm import tqdm

frameRate = 120.0

vec2 = pygame.math.Vector2
tf.compat.v1.disable_eager_execution()

game = Game()
possible_actions = np.identity(game.no_of_actions, dtype=int).tolist()

### MODEL HYPERPARAMETERS
state_size = [game.state_size]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = game.no_of_actions  # 9 possible actions
learning_rate = 0.00025  # Alpha (aka learning rate) # 0.00025

### TRAINING HYPERPARAMETERS
total_episodes = 50_000  # Total episodes for training
max_steps = 2000  # Max possible steps in an episode
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 7000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start, CHANGED THIS AS WE START WITH ADVANCED MODEL
explore_stop = 0.1  # minimum exploration probability
decay_rate = 0.0005 # exponential decay rate for exploration prob
gamma = 0.95  # Discounting rate

## GPU: 1million, CPU: 100_000
memory_size = 100_000

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
metaTraining = True
real_load_training_model_number = 0

if metaTraining:
    training =  True
    load = False
    starting_episode = real_load_training_model_number #This is considered in how many more episodes should be run
    load_training_model_number = real_load_training_model_number
else:
    training =  False
    load = True
    starting_episode = real_load_training_model_number
    load_traing_model = True
    load_training_model_number = real_load_training_model_number

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

tf.compat.v1.reset_default_graph()
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")
memory = Memory(memory_size)
game.new_episode()

clear_console()
print("pretraining")
for i in tqdm(range(memory_size)):
        if i == 0:
            state = game.get_state()

        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            next_state = np.zeros(state.shape)
            experience = state, action, reward, next_state, done
            memory.store(experience)
            game.new_episode()
            state = game.get_state()
        else:
            next_state = game.get_state()
            experience = state, action, reward, next_state, done
            memory.store(experience)
            state = next_state

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

print("Pretraining concluded...")
clear_console()
print("Training started")
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.Saver()
    if load:
        saver.restore(sess, "./allModels/model{}/models/model.ckpt".format(load_training_model_number))
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

    decay_step = real_load_training_model_number
    tau = 0
    game.new_episode()
    game.highScore = 0
    update_target = update_target_graph()
    sess.run(update_target)

    for episode in range(starting_episode, total_episodes):
        step = 0
        decay_step += 1
        episode_rewards = []
        game.new_episode()
        state = game.get_state()

        while step < max_steps:
            step += 1
            tau += 1
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
            reward = game.make_action(action)
            done = game.is_episode_finished()
            episode_rewards.append(reward)
            total_reward = np.sum(episode_rewards)
            game.highScore = max(game.get_score(), game.highScore)
            if done:
                next_state = np.zeros(state.shape, dtype=int)
                experience = state, action, reward, next_state, done
                memory.store(experience)
                step = max_steps
                if episode % 10 == 0:
                    print('Episode: {}'.format(episode) +
                        '\tTotal reward: {:.4f}'.format(np.sum(episode_rewards)/game.get_lifespan()) +
                        '\tExplore P: {:.4f}'.format(explore_probability) +
                        '\tScore: {}'.format(game.get_score()) +
                        '\tHighScore: {}'.format(game.highScore) +
                        '\tlifespan: {}'.format(game.get_lifespan())+ 
                        '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))) + "\n")

            else:
                next_state = game.get_state()
                experience = state, action, reward, next_state, done
                memory.store(experience)
                state = next_state

            ### LEARNING PART
            # Obtain random mini-batch from memory
            if step % 5 == 0:
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                states_mb = np.array([each[0][0] for each in batch], ndmin=2)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=2)
                dones_mb = np.array([each[0][4] for each in batch])
                target_Qs_batch = []
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    action = np.argmax(q_next_state[i])
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                                DQNetwork.target_Q: targets_mb,
                                                                DQNetwork.actions_: actions_mb,
                                                                DQNetwork.ISWeights_: ISWeights_mb})
                memory.batch_update(tree_idx, absolute_errors)

            if tau > max_tau:
                # Update the parameters of our TargetNetwork with DQN_weights
                update_target = update_target_graph()
                sess.run(update_target)
                tau = 0
                print(f"MODEL UPGRADE & WRITING TO TXT, {episode}")
                directory = "./allModels/model{}".format(episode)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = saver.save(sess, "./allModels/model{}/models/model.ckpt".format(episode))

                with open('QLearningFromOldMate.txt', 'a') as file:
                    file.write('Episode: {}'.format(episode) +
                    '\tTotal reward: {:.4f}'.format(total_reward) +
                    '\tExplore P: {:.4f}'.format(explore_probability) +
                    '\tScore: {}'.format(game.get_score()) +
                    '\tHighScore: {}'.format(game.highScore) +
                    '\tlifespan: {}'.format(game.get_lifespan())+ 
                    '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))) + "\n")