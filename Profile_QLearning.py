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
import cProfile, pstats

def main():
    frameRate = 30.0

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
    memory_size = 10000  # Number of experiences the Memory can keep #100000
    pretrain_length = memory_size  # Number of experiences stored in the Memory when initialized for the first time

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    metaTraining = True
    real_load_training_model_number = 175000

    if metaTraining:
        training =  True
        load = True #Do you want to start your training from a pre-existing model?
        starting_episode = real_load_training_model_number #This is considered in how many more episodes should be run
        #load_traing_model = False
        load_training_model_number = real_load_training_model_number
    else:
        training =  False
        load = True
        starting_episode = real_load_training_model_number
        load_traing_model = True
        load_training_model_number = real_load_training_model_number

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

    """ PRETRAIN """
    print("pretraining")
    for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                # First we need a state

                state = game.get_state()
                # state, stacked_frames = stack_frames(stacked_frames, state, True)

            # Random action
            action = random.choice(possible_actions)

            # Get the rewards
            reward = game.make_action(action)

            # Look if the episode is finished
            done = game.is_episode_finished()

            # If we're dead
            if done:
                # We finished the episode so the next state is just a blank screen
                next_state = np.zeros(state.shape)
                # print(state.shape)
                # Add experience to memory
                # experience = np.hstack((state, [action, reward], next_state, done))

                experience = state, action, reward, next_state, done
                memory.store(experience)

                # Start a new episode
                game.new_episode()

                # First we need a state
                state = game.get_state()


            else:
                # Get the next state
                next_state = game.get_state()

                # Add experience to memory
                experience = state, action, reward, next_state, done
                memory.store(experience)

                # Our state is now the next_state
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

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        if load:
            saver.restore(sess, "./allModels/model{}/models/model.ckpt".format(load_training_model_number))
        else:
            sess.run(tf.compat.v1.global_variables_initializer())

        decay_step = 0
        tau = 0

        # Init the game
        game.new_episode()
        game.highScore = 0

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(starting_episode, total_episodes):
            
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            game.new_episode()

            state = game.get_state()

            ## game.render() MISSING WINDOWS 

            while step < max_steps:
                step += 1
                tau += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                                possible_actions)

                reward = game.make_action(action)
                done = game.is_episode_finished()
                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros(state.shape, dtype=int)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    if game.get_score() > game.highScore:
                        game.highScore = game.get_score()

                    if episode % 1000 == 0:
                            print('Episode: {}'.format(episode),
                            '\tTotal reward: {:.4f}'.format(total_reward),
                            # '\tTraining loss: {:.4f}'.format(loss),
                            '\tExplore P: {:.4f}'.format(explore_probability),
                            '\tScore: {}'.format(game.get_score()),
                            '\tHighScore: {}'.format(game.highScore),
                            '\tlifespan: {}'.format(game.get_lifespan()),
                            '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))))

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    # Get the next state
                    next_state = game.get_state()

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
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
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                                DQNetwork.target_Q: targets_mb,
                                                                DQNetwork.actions_: actions_mb,
                                                                DQNetwork.ISWeights_: ISWeights_mb})
                if loss == 0:
                    print(ISWeights_mb)

                memory.batch_update(tree_idx, absolute_errors)

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("MODEL UPGRADE")

                    directory = "./allModels/model{}".format(episode)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    save_path = saver.save(sess, "./allModels/model{}/models/model.ckpt".format(episode))

                    print('Episode: {}'.format(episode) +
                        '\tTotal reward: {:.4f}'.format(total_reward) +
                        '\tExplore P: {:.4f}'.format(explore_probability) +
                        '\tScore: {}'.format(game.get_score()) +
                        '\tHighScore: {}'.format(game.highScore) +
                        '\tlifespan: {}'.format(game.get_lifespan())+ 
                        '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))) + "\n")

                    with open('QLearningFromOldMate.txt', 'a') as file:
                        print("WRITING TO TXT-File")
                        file.write('Episode: {}'.format(episode) +
                        '\tTotal reward: {:.4f}'.format(total_reward) +
                        '\tExplore P: {:.4f}'.format(explore_probability) +
                        '\tScore: {}'.format(game.get_score()) +
                        '\tHighScore: {}'.format(game.highScore) +
                        '\tlifespan: {}'.format(game.get_lifespan())+ 
                        '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))) + "\n")

print("We're about to run main")

def profile_script(entry_function):
    profiler = cProfile.Profile()
    profiler.runcall(entry_function)
    profiler.dump_stats('profile_stats.prof')

def is_user_defined(filename):
    return 'python' not in filename and 'lib' not in filename

def custom_filter():
    p = pstats.Stats('profile_stats.prof')
    p.strip_dirs()
    p.sort_stats('cumulative')
    for filename in p.stats:
        if is_user_defined(filename[0]):
            p.print_line(filename)

if __name__ == "__main__":
    profile_script(main)
    custom_filter()
