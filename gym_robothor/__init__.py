from gym.envs.registration import register

register(id='mtank_robothor-v0',
         entry_point='gym_robothor.envs:RoboThorEnv')
