from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import numpy as np



if __name__ == '__main__':
    env = AI2ThorEnv()
    n = env.action_space.n
    env.reset()
    episodes = []
    for i in range(5):
        env.reset()
        d = False
        total_r = 0.
        while not d:
            a = np.random.choice(n)
            o,r,d,_ = env.step(a)
            total_r +=r

        episodes.append(total_r)

        print(f'Total reward in episode {i} is {total_r}')

    print("AVG episode rewards:",episodes, np.mean(episodes))