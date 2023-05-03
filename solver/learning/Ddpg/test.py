

import gym

env = gym.make('CartPole-v1',render_mode="rgb_array")
observation = env.reset()

for t in range(1000):
    env.render()
    action = env.action_space.sample()
    step_result = env.step(action)
    observation, reward, done = step_result[:3]
    info = step_result[3]

    if done:
        observation = env.reset()

env.close()
