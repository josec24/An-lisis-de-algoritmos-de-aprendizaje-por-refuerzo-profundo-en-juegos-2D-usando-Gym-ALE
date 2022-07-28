from wrappers.wrappers import Wrappers
from models.A3C import A3C
import torch.optim as optim
import torch
import numpy as np
from agents.A3C_agent import A3CAgent

NAME, DEFAULT_ENV_NAME='Breakout', 'ALE/Breakout-v5'

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank=None, shared_model = None):
    
    env = Wrappers.make_env(DEFAULT_ENV_NAME)

    env.seed(8967 + rank * 5)
    
    model =A3C(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)
    episode = 0

    record=0

    agent = A3CAgent(env)

    # * Environment Setting before training * #
    observation = env.reset()
    state = observation
    lives = 5
    max_steps=10000
    max_episodes=10000

    for episode in range(max_episodes):
        model.load_state_dict(shared_model.state_dict())
        #logaritmo de la distribucion de politicas
        log_probs = []
        # valores
        values = []
        #recompensas
        rewards = []
        score=0
        #Estado inicial
        lives=5
        rewardo=0
        state = env.reset()

        for steps in range(max_steps):
            #obteniendo el valor y la política de distribución
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a)

            value, policy_dist = agent.actor_critic(state_v)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy()

            #Acción del agente
            action=agent.get_action(dist)

            #logaritmo de la distribucion de politicas
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            
            #Entropia para calcular la pérdida
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            new_state, reward, is_done, _ = env.step(action)

            if lives != _['lives']:
                #print(rewards)
                lives = _['lives']
                #print('resto que resto')
                reward -= 1
            
            rewardo+=reward

            #Guardando valores
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            agent.entropy_term += entropy

            score=rewardo

            #Si termina el paso
            if is_done or steps == max_steps-1 or _['lives']==0:

                break

        if episode % 1==0:
            new_state_a = np.array([new_state], copy=False)
            new_state_v = torch.tensor(new_state_a)

            #Obtener la perdida con los Q valores del modelo DQN y modelo objetivo DQN  
            Qval, _ = agent.actor_critic(new_state_v)
            Qval = Qval.detach().numpy()[0,0]

            ac_loss=agent.compute_loss(values[-1500:],rewards[-1500:],log_probs[-1500:],Qval)
            
            #Actualizar
            agent.ac_optimizer.zero_grad()
            ac_loss.backward()
            agent.ac_optimizer.step()

        ensure_shared_grads(model, shared_model)
        optimizer.step()