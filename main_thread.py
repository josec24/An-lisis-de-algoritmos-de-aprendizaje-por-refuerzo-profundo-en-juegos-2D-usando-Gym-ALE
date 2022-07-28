import torch
from models.A3C import A3C
import time
import numpy as np
from data.helper import guardarPuntuacion

def main_thread(env, shared_model):
    best_reward = 0
    episode = 0
    model = A3C(env.observation_space.shape, env.action_space.n)
    state = env.reset()
    done = True
    step = 0
    episode_reward = 0
    early_stop = False

    NAME, DEFAULT_ENV_NAME='Breakout', 'ALE/Breakout-v5'

    while True:
        model.load_state_dict(shared_model.state_dict())
        model.eval()
        
        while True:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a)
            value, policy_dist= model(state_v)

            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(env.action_space.n, p=np.squeeze(dist))
            

            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done or info['lives']==0:
                print("Episode finished after {} timesteps".format(step+1))
                if early_stop:
                    print(f"early stopping")

                print("game {},Best reward {}, episode reward {}".format( episode,best_reward,episode_reward))
                
                guardarPuntuacion(episode_reward,'A3C_'+NAME+'.csv')

                state = env.reset()
                    
                if best_reward < episode_reward and early_stop == False:
                    best_reward = episode_reward
                    print("game {} , new best reward {}, episode reward {}, saving model".format( episode,best_reward,episode_reward))
                    torch.save(model.state_dict(), './learnedModels/'+NAME+'_A3C_model.pt')
                    #* save model
                    
                episode += 1
                episode_reward = 0
                step = 0
                early_stop = False
                break
            
            state = new_state
            step += 1