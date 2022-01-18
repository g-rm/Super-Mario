import gym
import numpy as np
import retro
import time

from baselines.common.atari_wrappers import WarpFrame, FrameStack


def make_env(stack=True, record=False):
   
    '''Creates enviroment with custom wrappers'''

    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', 
                     players=1, record=record)
    
    env = MarioActions(env)
    env = CustomReward(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    
    return env


class MarioActions(gym.ActionWrapper):
    """
    Custom wrapper for Mario actions
    """
    def __init__(self, env):
        super(MarioActions, self).__init__(env)
        
        meaningful_actions = np.array([
        [0., 0., 0., 0., 0., 0., 0., 1., 0.],  # RIGHT
        [1., 0., 0., 0., 0., 0., 0., 1., 0.],  # B + RIGHT
        [0., 0., 0., 0., 1., 0., 0., 1., 1.],  # A + RIGHT 
        [0., 0., 0., 0., 0., 0., 0., 0., 1.],  # A
        [0., 0., 0., 0., 0., 0., 1., 0., 0.]   # LEFT
        ], dtype='?')

        action_meaning = None

        self._actions = []
        self._actions.extend(meaningful_actions)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._actual_Mario  = 0
        self._checkpoint    = 0
        self._currentX      = 0
        self._finalFlag     = 589824
        self._blockCounter  = 30
        self._timer         = True

    def reset(self):
        self._current_X     = 0
        self._blockCounter  = 30
        
        return self.env.reset()


    def step(self, action):
        state, reward, done, info = self.env.step(action)
        done    = info['endOfLevel']
        
        #death
        if not info['alive']:
            done = 1
            self.reset()

        else:

            #hit block reward
            if info['blockCounter'] < self._blockCounter:
                reward += 2500.0
                self._blockCounter = info['blockCounter']


            #mid-checkpoint game reward
            if info['checkpoint'] != self._checkpoint:
                reward += 4000.0
                self._checkpoint = info['checkpoint']


            #Mario powerUps (big, cape, fire) reward
            if info['powerups'] != self._actual_Mario:
                if info['powerups']:
                    reward += 3000.0
                else:
                    reward -= 1000.0

                self._actual_Mario = info['powerups']


            #walk to right reward
            if (info['x'] - self._currentX) > 100:
                reward += 300.0
                self._currentX = info['x']


            #punishes blocked agent
            if info['blocked'] == 5:
                reward -= 200.0


            #punishes stay quiet
            if info['timer1'] == 9:
                self._currentX = info['x']
                self._aux      = False

            if self._timer:
                self._start = time.time()
                self._currentX = info['x']
                self._timer = False

            if time.time() > self._start + 5:
                if (info['x'] - self._currentX) < 60:
                    reward -= 350.0
                    self._timer = True

            #rewards when grab bonus flag
            if info['endOfLevel']:
                if info['finalFlag'] == self._finalFlag:
                    reward += 7000.0
                else:
                    reward += 4500.0

                t = info['timer100'] * 10 + info['timer10']
                if t > 25:
                    reward += 10000.0

        self.env.render()

        return state, reward / 100.0, done, info
