"""
Different task implementations that can be defined inside an ai2thor environment
"""

from gym_robothor.utils import InvalidTaskParams
import numpy as np


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.max_episode_length = config.get('max_episode_length', 1000)
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """
        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class NavTask(BaseTask):
    def __init__(self, **kwargs):
        super(NavTask, self).__init__(kwargs)
        self.target_id = None # use id because target_obj is just a state and it must be updated as every time step
        self.pre_distance = 0
        self.num_collision = 0


    def transition_reward(self, event):
        reward, done = self.movement_reward, False

        #calculate the distance:
        assert self.target_id != None
        target_obj = event.get_object(self.target_id)
        agent_pos = [event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z']]
        target_pos = [target_obj['position']['x'], target_obj['position']['z']]

        dis = np.sqrt(np.sum(np.square(np.array(agent_pos)-np.array(target_pos))))
        
        if self.pre_distance > dis:
            print('Approaching destination')
        reward += (self.pre_distance - dis)

        # print('distance',(self.pre_distance-dis))
        # print(target_obj['name'])
        # print(target_obj['visible'])

        # collision
        if event.metadata['lastActionSuccess'] == False:
            print('Collision!!!')
            reward -= 1.0 # this should be large, so that make the movement reward more worthwhile.
            self.num_collision += 1
            # done = True

        # reach destination
        if dis < 1.1 and target_obj['visible']:
            reward += 10.0
            done = True
            print('>>>>>>>>>>>>>>>> Reached destination')
        
        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            print('{} meters from destination'.format(dis))
            done = True
        
        # print('reward', reward)
        
        self.step_num += 1
        self.pre_distance = dis

        return reward, done

    def reset(self):
        self.step_num = 0
        self.num_collision = 0

       


