#!/usr/bin/env python

import tensorflow as tf
import os
import time
import argparse

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from mario_util import make_env


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--restore', '-restore', 
                        action='store_true', 
                        help='restore from checkpoint file')
    
    args = parser.parse_args()
    checkpoint_dir  = os.path.join(os.getcwd(), 'results')
    
    #Creates environment
    env = make_env(stack=True, record=False)
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    #configure hardware
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                env.action_space.n,
                gym_space_vectorizer(env.observation_space),
                min_val=-20,
                max_val=70))

        saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=1.0)
        if args.restore:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                saver.restore(sess, latest_checkpoint)
            else:
                print("Checkpoint not found")        

        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 5)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())


        #play the game!
        done=0
        while(not done):
            transitions = player.play()
            for trans in transitions:
                if trans['is_last']:
                    done=1


if __name__ == '__main__':
    try:
        main()
    except:
        print("end of trainning")
