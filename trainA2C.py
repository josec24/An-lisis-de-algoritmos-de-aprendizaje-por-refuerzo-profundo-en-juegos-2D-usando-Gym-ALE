import torch.optim as optim
from wrappers.wrappers import Wrappers
import numpy as np
import torch
from agents.A2C_agent import A2CAgent

from data.helper import guardarPuntuacion

#Entrenamiento del agente
def train(env, agent, max_episodes, max_steps):
    
    record = 2
    score=0
    mean_score=0
    # Recorriendo cada episodio
    for episode in range(max_episodes):

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
                lives = _['lives']
                reward -= 1
            
            rewardo+=reward

            #Guardando valores
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            agent.entropy_term += entropy
            #Siguiente estado

            score=rewardo

            #Si termina el paso
            if is_done or steps == max_steps-1 or _['lives']==0:
                agent.all_rewards.append(rewardo)
                
                mean_score = np.mean(agent.all_rewards[-100:])
                
                if score > record:
                    record = score
                    
                    print("Game: {}, New best reward {}, best reward {}, mean score {}, saving model...".format(episode, score, record,mean_score))
                    torch.save(agent.actor_critic.state_dict(), './learnedModels/'+NAME+'_A2C_model.pt')
                
                else:
                    print("Game: {}, reward {}, best reward {}, mean score {}.".format(episode, score, record,mean_score))
                
                guardarPuntuacion(score,'A2C_SpaceInvaders.csv')

                break
            
        #Se puede cambiar aqui la frecuencia de actualizacion
        if episode % 1==0:
            new_state_a = np.array([new_state], copy=False)
            new_state_v = torch.tensor(new_state_a)

            Qval, _ = agent.actor_critic(new_state_v)
            Qval = Qval.detach().numpy()[0,0]

            ac_loss=agent.compute_loss(values[-1500:],rewards[-1500:],log_probs[-1500:],Qval)
            
            #Actualizar
            agent.ac_optimizer.zero_grad()
            ac_loss.backward()
            agent.ac_optimizer.step()


#Episodios máximos
MAX_EPISODES = 100000
#Pasos máximos
MAX_STEPS = 100000
NAME, DEFAULT_ENV_NAME='SpaceInvaders', 'ALE/SpaceInvaders-v5'
#Entorno(juego)
env = Wrappers.make_env(DEFAULT_ENV_NAME)
#Agente
agent = A2CAgent(env)
#Entrenando
train(env, agent, MAX_EPISODES, MAX_STEPS)