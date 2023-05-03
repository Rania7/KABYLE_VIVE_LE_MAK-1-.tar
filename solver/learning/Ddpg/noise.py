import numpy as np
# Ornstein_Uhlenbeck_process
# le processus d'Ornstein-Uhlenbeck est utilisé pour 
# ajouter du bruit d'exploration à l'espace d'action.
#

#Le réseau acteur est utilisé pour sélectionner 
#des actions en fonction de l'état actuel de 
#l'environnement, et du bruit d'exploration est 
#ajouté aux actions pour encourager l'exploration de 
#l'espace d'action.

#Le processus d'Ornstein-Uhlenbeck est un choix
#populaire pour générer du bruit d'exploration 
#dans DDPG car il produit un bruit corrélé qui 
#convient bien aux problèmes de contrôle continu. 
 

 
# mu mean for the noise
# sigma standard deviation 
# theta 
# dt = time parameter
#  x0 = starting value
class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


