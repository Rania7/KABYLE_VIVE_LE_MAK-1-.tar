# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..net import GATConvNet, GCNConvNet, ResNetBlock, MLPNet


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
        logits, outputs, hidden_state = self.actor(obs)
        self._last_hidden_state = hidden_state
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

        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)

    def forward(self, obs, action):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)
        return value


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
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # shape: (batch_size, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # shape: (batch_size, 1, hidden_dim * num_directions)
        return context, attn_weights