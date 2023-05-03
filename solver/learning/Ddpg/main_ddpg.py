import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    #print("env :",env,"\n\n")

    print("env.observation_space.shape",env.observation_space.shape)
    print("nv.action_space.shape",env.action_space.shape[0])


    # alpha 0.0001

    ##### state contains ?

    agent = Agent(alpha=0.0001, beta=0.01, 
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    #print("env:env.observation_space.shape",env.observation_space.shape)
    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    
    best_score = env.reward_range[0]
    score_history = []


    for i in range(n_games):
        #print("i",i)
        
        observation, info =env.reset(seed=0) #seed=42

        #print("env : observation",observation)
        terminated = False
        truncated =False
        score = 0
        agent.noise.reset()
        while not terminated and not truncated :
            action = agent.choose_action(observation)

            #terminated=True if environment terminates (eg. due to task completion, 
            #failure etc.) truncated=True if episode truncates due to a time limit or a 
            #reason that is not defined as part of the task MDP.
            
            #print("observation",observation)
            #print("action",action)

            observation_, reward,  terminated, truncated, info = env.step(action)  ############## step???


            agent.remember(observation, action, reward, observation_, terminated)
            agent.learn()
            
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)




