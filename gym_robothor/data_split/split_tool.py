import ai2thor.controller
import ai2thor.util.metrics
import json
from gym_robothor.utils import read_config
import time
import numpy as np


def get_idpos_through_type(event, object_type):
    for obj in event.metadata['objects']:
        if obj['objectType'] == object_type:
            # print('>>>>>>>>>>>>>>>>>>>',obj['objectId'], obj['position'])
            return obj['objectId'], obj['position']
    print('Over......................')
    all_visible_objects = [obj['objectType'] for obj in event.metadata['objects']]
    print(all_visible_objects)


class Split_tool:
    def __init__(self):

        self.config = read_config('config_files/NavTaskTrain.json')
        self.split_valid = []
        self.split_loss = []
        # self.controller = ai2thor.controller.Controller(agentMode='bot')
        with open('gym_robothor/data_split/train_valid_.json') as f:
            self.episodes = json.loads(f.read())
        
        self.loss = 0
        self.valid = 0

        self.max = 100000000

        

    def run(self):
        for i, e in enumerate(self.episodes):
            if i < self.max:
            #     print("Task Start id:{id} scene:{scene} target_object:{object_id} initial_position:{initial_position} rotation:{initial_orientation}".format(**e))
            #     self.controller.initialization_parameters['robothorChallengeEpisodeId'] = e['id']
            #     self.controller.reset(e['scene'])
            #     teleport_action = dict(action='TeleportFull')
            #     teleport_action.update(e['initial_position'])
            #     event = self.controller.step(action=teleport_action)
            #     event = self.controller.step(action=dict(action='Rotate', rotation=dict(y=e['initial_orientation'], horizon=0.0)))  
            #     # obj_type =  e['object_type'].replace(' ','') if ' ' in e['object_type'] else e['object_type']
            #     # e['object_id'], e['target_position'] = get_idpos_through_type(event, obj_type)
            #     # print('<<<<<<<<<<<<<<<<<',e['object_id'])
            #     try:
            #         path = ai2thor.util.metrics.get_shortest_path_to_object(self.controller, e['object_id'], e['initial_position'], e['initial_orientation'])
            #         dis = ai2thor.util.metrics.path_distance(path)
            #         e['shortest_path'] = path
            #         e['shortest_path_length'] = dis
            #         self.split_valid.append(e)
            #         self.valid += 1
            #     except:
            #         self.loss += 1
            #         self.split_loss.append(e)
            # else:
            #     break
                
                # filter dis
                vector = np.array([e['initial_position']['x'], e['initial_position']['z']]) - \
                    np.array([e['target_position']['x'], e['target_position']['z']])
                # dis =  np.sqrt(np.sum(np.square(vector)))
                if vector[0] > 2 and vector[1] > 2:
                    self.split_valid.append(e)
                    self.valid += 1

                # print(path, "<<<<<<<<<<<<<", dis)
                # time.sleep(5)

        with open('./gym_robothor/data_split/train_valid_dis.json','w') as f:
            json.dump(self.split_valid, f, indent=4)
        
        # with open('test_loss_.json','w') as f:
        #     json.dump(self.split_loss, f, indent=4)

        print('Result: Valid:{}, Loss:{}, rate:{}'.format(self.valid, self.loss, self.valid/len(self.episodes)))

if __name__ == '__main__':
    tool = Split_tool()
    tool.run()


# Result: Valid:19058, Loss:6016, rate:0.7600701922309963 training
# Result: Valid:4127, Loss:2163, rate:0.656120826709062

# Result: Valid:27330, Loss:265, rate:0.9903968110164885
# Result: Valid:5645, Loss:471, rate:0.9229888816219751