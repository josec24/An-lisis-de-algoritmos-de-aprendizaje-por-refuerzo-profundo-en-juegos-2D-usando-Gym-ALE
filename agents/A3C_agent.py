from models.A3C import A3C
import torch.optim as optim
import numpy as np
import torch

class A3CAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma=gamma

        self.actor_critic = A3C(env.observation_space.shape, env.action_space.n)

        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.all_rewards = []
        self.entropy_term = 0

    #Obtener la acción(devuelve el número que indica la acción)
    def get_action(self,dist):
        action = np.random.choice(self.env.action_space.n, p=np.squeeze(dist))
            
        return action

    #Obtener la perdida con los Q valores del modelo DQN y modelo objetivo DQN
    def compute_loss(self, values,rewards,log_probs,Qval):     
        # Valores Q
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + self.gamma * Qval
            Qvals[t] = Qval
    
        #Actualizar actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
            
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term
        
        return ac_loss