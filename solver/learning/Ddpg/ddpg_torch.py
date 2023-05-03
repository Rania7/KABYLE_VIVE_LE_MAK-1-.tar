import os
import numpy as np
import torch as T
import torch.nn.functional as F
from net import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer




class Agent():
    #learning rate alpha and beta
    # hyper param tau
    #gamma for agent
    #max_size for buffer
    #number of neuronne for frst and second layer
    #batch size for replay memory
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha 
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        #mu mean of distribution 
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor')


        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic')
        
        #hard copy from main network to target
        self.update_network_parameters(tau=1)




    def choose_action(self, observation):
        #set the actor to evaluate
        #if not doing batch norm or drop out in network=> dont' need avaluting the actor
        self.actor.eval() #### #!###########################################################eval ???????????????????????????
        ##me
        #print("1",observation)
        #print("Observation shape:", observation.shape)
        #observation = np.array(observation)
        #print("2",observation)
        #print(shape(observation))
        #observation=observation.reshape((2,8))
        ##

        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        
        #totaly deterministic proba distribution
        mu = self.actor.forward(state).to(self.actor.device)
        #print("##############  mu",mu)


        # add noise
        mu_prime = mu + T.tensor(self.noise(), 
                                  dtype=T.float).to(self.actor.device)
        

        # dans cette phase il réalise des opération (dropout regularization,
        #batch normalization)
        self.actor.train()####################################################################################???????????
        #env takes numpy array so we detach
        return mu_prime.cpu().detach().numpy()[0]


    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)




    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()




    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()




    ###################learning agent#########################
    def learn(self):
        
        #if we don't have at least batch_size of transitions we
        # came back to the main
        if self.memory.mem_cntr < self.batch_size:
            return
        ###########sampling the buffer
        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        #boolean arrays
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)


        ###########pass the tensors through the networks
        
        target_actions = self.target_actor.forward(states_)#1
        critic_value_ = self.target_critic.forward(states_, target_actions)#2 
        
        critic_value = self.critic.forward(states, actions) #true value
        
        #affecte 0.0 aux elements ayant la var done =TRUE
        critic_value_[done] = 0.0  ##########################??
        # from tensor to 1Dim tensor(vector)
        critic_value_ = critic_value_.view(-1)
        
        #######################################Goal

        #yi = ri + gamma (discount factor) * target_critic_Q(state_future, target_actor(state_future)#1)#2

        target = rewards + self.gamma*critic_value_

        target = target.view(self.batch_size, 1) # from tensor to tensor 2 D

        self.critic.optimizer.zero_grad()#mettre à 0 les gradients du RN critic avant le calcul des gradients (backward)
        

        critic_loss = F.mse_loss(target,critic_value)   

        critic_loss.backward() ############# code 


        #update critic network
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()


        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)

        actor_loss.backward()
        #update actor
        self.actor.optimizer.step()

        self.update_network_parameters()




  ##################UPDATE TARGET NETWORK#######################
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
      # get name params, copy the param to dict, modify by calling
      #the values of param with param[name param] since dict
      # ==>  upload

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()    
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        
        #modify in dict
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()
        
        # load to target
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

        
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)





