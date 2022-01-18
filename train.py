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

REWARD_HISTORY = 100

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--restore', '-restore', 
                        action='store_true', 
                        help='restore from checkpoint file')
    

    args = parser.parse_args()
    checkpoint_dir  = os.path.join(os.getcwd(), 'results')
    results_dir     = os.path.join(os.getcwd(), 'results', 
                                  time.strftime("%d-%m-%Y_%H-%M-%S"))
    

    summary_writer = tf.summary.FileWriter(results_dir)
    
    #Creates environment
    env = make_env(stack=True, record=False)
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4)

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

        reward_hist = []
        model_hist  = []
        total_steps = 0

        # runs with every completed episode
        def _handle_ep(steps, rew):
            nonlocal total_steps, model_hist
            total_steps += steps
            reward_hist.append(rew)
            
            summary_reward = tf.Summary()
            summary_reward.value.add(tag='global/reward', simple_value=rew)
            summary_writer.add_summary(summary_reward, global_step=total_steps)

            print('Steps: ', total_steps, 'Reward: ', rew)
            print('save model\n')
            saver.save(sess=sess, save_path=checkpoint_dir + '/model', 
                                  global_step=total_steps)
           
            model_number = '/model-' + str(total_steps)
            model_hist.append(model_number)

            if len(reward_hist) == REWARD_HISTORY:
                print('%d steps: mean=%f' % (total_steps, 
                                            sum(reward_hist) / len(reward_hist)))
                
                for model in model_hist[:-1]:
                    try:
                        os.remove(checkpoint_dir + model + '.data-00000-of-00001')
                        os.remove(checkpoint_dir + model + '.meta')
                        os.remove(checkpoint_dir + model + '.index')
                    except:
                        print('Model not found')

                model_hist.clear()
                reward_hist.clear()

        dqn.train(num_steps=3500000, 
                player=player,
                replay_buffer=PrioritizedReplayBuffer(300000, 0.5, 0.4, epsilon=0.1),
                optimize_op=optimize,
                train_interval=3,
                target_interval=8192,
                batch_size=32,
                min_buffer_size=200000,
                handle_ep=_handle_ep)

if __name__ == '__main__':
    try:
        main()
    except:
        print("end of trainning")
