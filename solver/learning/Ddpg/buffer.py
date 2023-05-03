import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size

        self.mem_cntr = 0 #know the available memory

        self.state_memory = np.zeros((self.mem_size, *input_shape))  #####? 
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))


        # actions are continuous number, so it's not the number of action
        # mais le nombre de composantes de l'action
        self.action_memory = np.zeros((self.mem_size, n_actions))
        
        self.reward_memory = np.zeros(self.mem_size)
        
        #used as mask to set the critic values for the new state 
        # to zero 0
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.truncated_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, state_, terminated):
        #si on arrive à la fin du buffer, on reprend des le début 
        #avec le modulos
        # si meme =12 , on est à 13 => donc on se place à 1
        #car le mem_counter (cntr) ne fait qu'augmenter

        index = self.mem_cntr % self.mem_size #modulos   
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminated

        self.mem_cntr += 1



    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)

        # random bounch (list) between  max_memory and batch_size

        batch = np.random.choice(max_mem, batch_size)  #####  batch_size element in (0,max_mem)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminates = self.terminal_memory[batch]

        return states, actions, rewards, states_,terminates


