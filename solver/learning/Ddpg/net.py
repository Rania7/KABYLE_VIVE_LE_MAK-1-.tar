# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..net import GATConvNet, GCNConvNet, ResNetBlock, MLPNet
import numpy as np
import torch.optim as optim


# bleu detail, last hidden-state ??
class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.encoder = Encoder(v_net_feature_dim, embedding_dim=embedding_dim)

        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)

        self.target_actor=Actor(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self.target_critic= Critic(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)




        self._last_hidden_state = None

    def encode(self, obs): #encode?
        x = obs['v_net_x']  # node data
        outputs, hidden_state = self.encoder(x)
        self._last_hidden_state = hidden_state
        print("i am in net, _last_hidden_state",hidden_state)
        return outputs

    def act(self, obs):
        print("_____________hello____________")
        print(obs)
        #logits, outputs, hidden_state = self.actor(obs)
        logits = self.actor(obs)

        #self._last_hidden_state = hidden_state
        print("i am in net, logits",logits)

        return logits

    def evaluate(self, obs, action):  # evaluate doit prendre obs and action dans la ddpg Q(s,a) not V(s) comme dans le cas A3C
        value = self.critic(obs, action)
        return value

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(Actor, self).__init__()
        '''
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)
        '''
        #__init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions)
        #to start 
        
        alpha = 0.001
        input_dims=300
        fc1_dims=120
        fc2_dims=64
        n_actions=p_net_num_nodes 




        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = T.nn.Linear(self.input_dims, self.fc1_dims) # *self.input_dims
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        eps = 1e-5

        self.bn1 = nn.LayerNorm(self.fc1_dims,eps)
        self.bn2 = nn.LayerNorm(self.fc2_dims,eps)

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

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # SGD optmizer
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
        

        


    def forward(self, obs):

        """Return logits of actions"""

        '''
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state'''
        file_v_v_net= open("forward_actor_done_fc1.txt","w")
        file_v_v_net.write(str(obs)+"----input----\n\n")



        x = self.fc1(obs)
        file_v_v_net.write(str(x)+"----output----\n\n")
        file_v_v_net.close()

        #eps = 1e-5
        #x=self.bn1(x)
        file_v_v_net= open("forward_actor_done_relu1.txt","w")

        file_v_v_net.write(str(x)+"----input----\n\n")

        x = F.relu(x)
        file_v_v_net.write(str(x)+"----output----\n\n")
        file_v_v_net.close()
        

        file_v_v_net= open("forward_actor_done_fc2.txt","w")
        file_v_v_net.write(str(x)+"----input----\n\n")


        x = self.fc2(x)

        file_v_v_net.write(str(x)+"----output----\n\n")
        file_v_v_net.close()



        #x=self.bn2(x)
        file_v_v_net= open("forward_actor_done_relu2.txt","w")
        file_v_v_net.write(str(x)+"----input----\n\n")
        x = F.relu(x)  ####??
        file_v_v_net.write(str(x)+"----output----\n\n")
        file_v_v_net.close()
        # Tanh used because it gives the output between -1 and +1
        
        f=open("Tanh.txt","w")
        f.write(str(x)+"----input----\n\n")
        x = T.tanh(self.mu(x)) #?????? 
        
        # and it is convenient for the env in this cas 
        # so if in cas -2 +2 , multiply the outputs by the bounds
        # needed
        
        f.write(str(x)+"----output----\n\n")

        f.close()
        
        return x


class Critic(nn.Module):
    # learning rate beta
    # input_dims dimensions for fully connected layer
    # number of action
    #name for the check model 
    def __init__(self, obs, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(Critic, self).__init__()
        '''
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)
        '''
        #    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):

        alpha = 0.001
        input_dims=300
        fc1_dims=120
        fc2_dims=64
        n_actions=p_net_num_nodes 

      
        beta=0.005

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = 50


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) #        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

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






    def forward(self, obs, action):
        """Return logits of actions"""
        '''
        logits, outputs, hidden_state = self.decoder(obs)
        value = T.mean(logits, dim=-1, keepdim=True)
        return value
        '''
        

        state_value = self.fc1(obs)
        #on normalise avant car si y des valeurs negatif
        # et elle ont un sens exp (direction pour la vitesse)
        #si on fait relu avant normalisation
        # on perdra l'info 
        #state_value = self.bn1(state_value) 
        # relu supprime tt ce qui est negatif
        state_value = F.relu(state_value)
         
        state_value = self.fc2(state_value)
        #state_value = self.bn2(state_value)
        #state_value = F.relu(state_value)
        #action_value = F.relu(self.action_value(action))
        

        #action = action + [0] * (self.n_actions- len(action)+1)
        f=open("action.txt","w")
        f.write(str(action))
        f.close()


        #arr_long = action.astype(np.int64)

        # convert the long numpy array to a PyTorch tensor

        action=T.tensor(action, dtype=T.float).to("cpu").unsqueeze(dim=0)
        action_value = self.action_value(action)  # me a vérifier
        #joint
        state_action_value = F.relu(T.add(state_value, action_value))
        #state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value


class Encoder(nn.Module):

    def __init__(self, v_net_feature_dim, embedding_dim=64):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(v_net_feature_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embeddings = F.relu(self.emb(x))
        outputs, hidden_state = self.gru(embeddings)
        return outputs, hidden_state
    

class Decoder(nn.Module):
    
    def __init__(self, p_net_num_nodes, feature_dim, embedding_dim=64):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(p_net_num_nodes + 1, embedding_dim)
        self.att = Attention(embedding_dim)
        self.gcn = GCNConvNet(feature_dim, embedding_dim, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )
        # self.out = nn.Sequential(
        #     GCNConvNet(embedding_dim, 1, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True),
        #     nn.Flatten(),
        # )
        self.gru = nn.GRU(embedding_dim, embedding_dim)
        self._last_hidden_state = None

    def forward(self, obs):
        batch_p_net = obs['p_net']
        hidden_state = obs['hidden_state']
        p_node_embeddings = self.gcn(batch_p_net)
        p_node_embeddings = p_node_embeddings.reshape(batch_p_net.num_graphs, -1, p_node_embeddings.shape[-1])
        p_node_embeddings = p_node_embeddings + hidden_state
        logits = self.mlp(p_node_embeddings)
        p_node_id = obs['p_node_id']
        hidden_state = hidden_state.permute(1, 0, 2)
        encoder_outputs = obs['encoder_outputs']
        mask = obs['mask']
        p_node_emb = self.emb(p_node_id).unsqueeze(0)
        context, attention = self.att(hidden_state, encoder_outputs, mask)
        context = context.unsqueeze(0)
        outputs, hidden_state = self.gru(p_node_emb, hidden_state)
        return logits, outputs, hidden_state
    

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim * num_directions)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.transpose(0, 1).repeat(1, seq_len, 1)  # shape: (batch_size, seq_len, hidden_dim)
        energy = T.tanh(self.attn(T.cat([hidden, encoder_outputs], dim=2)))  # shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # shape: (batch_size, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        context = T.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # shape: (batch_size, 1, hidden_dim * num_directions)
        return context, attn_weights