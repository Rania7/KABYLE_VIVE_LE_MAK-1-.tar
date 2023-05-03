import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#state +action en entrée

class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()

        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self._last_hidden_state = None
        #### me 
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
       ### me 
    '''
    def encode(self, obs): #encode?
        x = obs['v_net_x']  # node data
        outputs, hidden_state = self.encoder(x)
        self._last_hidden_state = hidden_state
        print("i am in net, _last_hidden_state",hidden_state)
        return outputs
    '''
    def act(self, obs):
        logits, outputs, hidden_state = self.actor(obs)
        self._last_hidden_state = hidden_state
        print("i am in net, logits",logits)
        return logits

    def evaluate(self, obs):
        value = self.critic(obs)
        return value

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state

    
class Critic(nn.Module):
    # learning rate beta
    # input_dims dimensions for fully connected layer
    # number of action
    #name for the check model 
     
    
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # normalize les sorties des couches 1 et 2 of diffrent scales
        #goal => model can deal with multiple environments 
        # here layer normalization is used => beacause 
        #1/ independent from the batch size contrairement au batch norm
        #2/ when wopy the parameters from the main (regular) network to target => batch norm don't keep track of running mean and running variance => cause error
        #3 little faster
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)


        # the action is not include until the second layer
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        #layer to join evrything, to get the actual critic value
        self.q = nn.Linear(self.fc2_dims, 1)
         
        
        #get square root of the number of input dimensions
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        #set the weight to be a uniform distribution 
        self.fc1.weight.data.uniform_(-f1, f1)
        #same for biased data
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        
        # set the action_value layer (it works without doing it)
        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)
        
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # send into the device
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        #on normalise avant car si y des valeurs negatif
        # et elle ont un sens exp (direction pour la vitesse)
        #si on fait relu avant normalisation
        # on perdra l'info 
        state_value = self.bn1(state_value) 
        # relu supprime tt ce qui est negatif
        state_value = F.relu(state_value)
         
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        #state_value = F.relu(state_value)
        #action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        #joint
        state_action_value = F.relu(T.add(state_value, action_value))
        #state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)



# state en entrée
class Actor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)  
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)  ####??
        # Tanh used because it gives the output between -1 and +1
        

        x = T.tanh(self.mu(x)) #?????? 
        # and it is convenient for the env in this cas 
        # so if in cas -2 +2 , multiply the outputs by the bounds
        # needed
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
